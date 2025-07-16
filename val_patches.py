import os
import argparse
import wandb
import pyiqa
import numpy as np
from PIL import Image 
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from accelerate.utils import set_seed
from terediff.utils.common import instantiate_from_config, text_to_image
from terediff.model import ControlLDM, Diffusion
from terediff.sampler import SpacedSampler
import initialize

import math

def split_image_with_overlap(image, patch_size=128, overlap=16):
    """
    将图像分割成带有重叠的小块
    
    Args:
        image: PIL Image对象
        patch_size: 小块的尺寸 (默认128)
        overlap: 重叠的像素数 (默认16)
    
    Returns:
        list: 包含所有小块图像的列表，从左到右，自上而下排列
    """
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 获取图像尺寸
    if len(img_array.shape) == 3:
        height, width, channels = img_array.shape
    else:
        height, width = img_array.shape
        channels = 1
        img_array = img_array[:, :, np.newaxis]
    
    # 计算步长
    stride = patch_size - overlap
    
    # 计算需要的padding
    # 计算在每个方向上需要多少个patch
    num_patches_h = math.ceil((height - overlap) / stride)
    num_patches_w = math.ceil((width - overlap) / stride)
    
    # 计算padding后的尺寸
    padded_height = (num_patches_h - 1) * stride + patch_size
    padded_width = (num_patches_w - 1) * stride + patch_size
    
    # 计算需要的padding
    pad_h = padded_height - height
    pad_w = padded_width - width
    
    # 进行padding (在右边和下边padding)
    if len(img_array.shape) == 3:
        padded_img = np.pad(img_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    else:
        padded_img = np.pad(img_array, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    # 提取小块
    patches = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # 计算当前patch的起始位置
            start_h = i * stride
            start_w = j * stride
            
            # 提取patch
            if len(padded_img.shape) == 3:
                patch = padded_img[start_h:start_h + patch_size, start_w:start_w + patch_size, :]
                # 如果是单通道，去掉最后一个维度
                if channels == 1:
                    patch = patch[:, :, 0]
            else:
                patch = padded_img[start_h:start_h + patch_size, start_w:start_w + patch_size]
            
            # 转换回PIL Image
            patch_image = Image.fromarray(patch.astype(np.uint8))
            patches.append(patch_image)
    
    return patches

# 使用示例
def process_images_with_patches(gt_img_path, lq_img_path, patch_size=128, overlap=16):
    """
    处理图像并分割成小块的示例函数
    """
    gt_img = Image.open(gt_img_path)
    lq_img = Image.open(lq_img_path)
    
    # 分割GT图像
    gt_patches = split_image_with_overlap(gt_img, patch_size, overlap)
    
    # 分割LQ图像
    lq_patches = split_image_with_overlap(lq_img, patch_size, overlap)
    
    print(f"GT图像分割成 {len(gt_patches)} 个小块")
    print(f"LQ图像分割成 {len(lq_patches)} 个小块")
    
    return gt_patches, lq_patches


def merge_patches_with_overlap(patches, original_size, patch_size=512, overlap=64):
    """
    将带有重叠的图像块合并成一个大图
    
    Args:
        patches: list of torch.Tensor, 每个tensor形状为 (1, 3, 512, 512)
        original_size: tuple, 原始图像的尺寸 (height, width)
        patch_size: int, 小块的尺寸 (默认512)
        overlap: int, 重叠的像素数 (默认64)
    
    Returns:
        torch.Tensor: 合并后的图像tensor，形状为 (1, 3, height, width)
    """
    device = patches[0].device
    dtype = patches[0].dtype
    
    # 计算步长
    stride = patch_size - overlap
    
    # 计算原始图像需要的padding (基于128x128的patch计算)
    original_stride = 128 - 16  # 原始的stride
    original_height, original_width = original_size
    
    # 计算patch的数量
    num_patches_h = math.ceil((original_height - 16) / original_stride)
    num_patches_w = math.ceil((original_width - 16) / original_stride)
    
    # 计算padding后的尺寸 (基于原始128x128)
    padded_height = (num_patches_h - 1) * original_stride + 128
    padded_width = (num_patches_w - 1) * original_stride + 128
    
    # 现在需要将这个尺寸按比例放大到512x512
    scale_factor = patch_size / 128  # 4倍放大
    final_height = int(padded_height * scale_factor)
    final_width = int(padded_width * scale_factor)
    
    # 创建输出图像和权重图像
    merged_image = torch.zeros((1, 3, final_height, final_width), device=device, dtype=dtype)
    weight_map = torch.zeros((1, 1, final_height, final_width), device=device, dtype=dtype)
    
    # 创建权重窗口 (用于处理重叠区域)
    window = torch.ones((patch_size, patch_size), device=device, dtype=dtype)
    
    # 对重叠区域使用渐变权重
    fade_size = overlap
    for i in range(fade_size):
        # 上边缘
        window[i, :] *= (i + 1) / fade_size
        # 下边缘
        window[-(i+1), :] *= (i + 1) / fade_size
        # 左边缘
        window[:, i] *= (i + 1) / fade_size
        # 右边缘
        window[:, -(i+1)] *= (i + 1) / fade_size
    
    # 将每个patch放置到正确的位置
    patch_idx = 0
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if patch_idx >= len(patches):
                break
                
            # 计算当前patch在最终图像中的位置
            start_h = i * stride
            start_w = j * stride
            end_h = start_h + patch_size
            end_w = start_w + patch_size
            
            # 获取当前patch
            current_patch = patches[patch_idx]  # (1, 3, 512, 512)
            
            # 应用权重
            weighted_patch = current_patch * window.unsqueeze(0).unsqueeze(0)
            
            # 累加到输出图像
            merged_image[:, :, start_h:end_h, start_w:end_w] += weighted_patch
            weight_map[:, :, start_h:end_h, start_w:end_w] += window.unsqueeze(0).unsqueeze(0)
            
            patch_idx += 1
        
        if patch_idx >= len(patches):
            break
    
    # 归一化 (避免除零)
    weight_map = torch.clamp(weight_map, min=1e-8)
    merged_image = merged_image / weight_map
    
    # 裁剪到原始尺寸的放大版本
    target_height = int(original_height * scale_factor)
    target_width = int(original_width * scale_factor)
    merged_image = merged_image[:, :, :target_height, :target_width]
    
    return merged_image


def main(args):


    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(25, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    cfg = OmegaConf.load(args.config)
    
    # setup logging tool
    if cfg.log_args.log_tool == 'wandb':
        wandb.login(key=cfg.log_args.wandb_key)
        wandb.init(project=cfg.log_args.wandb_proj_name, 
                name='VAL_terediff_stage3_DEMO',
                config=argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))
        )

    
    # load demo images from demo_imgs/ folder
    gt_imgs_path = sorted([f"{cfg.dataset.gt_img_path}/{img}" for img in os.listdir(cfg.dataset.gt_img_path) if img.endswith((".jpg", ".jpeg", ".png"))])
    lq_imgs_path = sorted([f"{cfg.dataset.lq_img_path}/{img}" for img in os.listdir(cfg.dataset.lq_img_path) if img.endswith((".jpg", ".jpeg", ".png"))])

                    
    # load models
    models, _ = initialize.load_model(accelerator, device, args, cfg)
    

    # setup ddpm
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(device)
    sampler = SpacedSampler(diffusion.betas, diffusion.parameterization, rescale_cfg=False)


    # setup model accelerator    
    models = {k: accelerator.prepare(v) for k, v in models.items()}


    # unwrap cldm from accelerator for proper model saving
    pure_cldm: ControlLDM = accelerator.unwrap_model(models['cldm'])


    # SR metrics
    metric_psnr = pyiqa.create_metric('psnr', device=device)
    metric_ssim = pyiqa.create_metric('ssimc', device=device)
    metric_lpips = pyiqa.create_metric('lpips', device=device)
    metric_dists = pyiqa.create_metric('dists', device=device)
    metric_niqe = pyiqa.create_metric('niqe', device=device)
    metric_musiq = pyiqa.create_metric('musiq', device=device)
    metric_maniqa = pyiqa.create_metric('maniqa', device=device)
    metric_clipiqa = pyiqa.create_metric('clipiqa', device=device)


    tot_val_psnr=[]
    tot_val_ssim=[]
    tot_val_lpips=[]
    tot_val_dists=[]
    tot_val_niqe=[]
    tot_val_musiq=[]
    tot_val_maniqa=[]
    tot_val_clipiqa=[]
    

    # set seed for identical generation for validation sampling noise
    gen.manual_seed(25)
    
    
    # put model on eval
    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()


    # For val_gt (range [-1, 1])
    preprocess_gt = T.Compose([
        T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # For val_lq (range [0, 1])
    preprocess_lq = T.Compose([
        T.Resize(size=(512, 512), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    for val_batch_idx, (gt_img_path, lq_img_path) in enumerate(tqdm(zip(gt_imgs_path, lq_imgs_path), desc='val', total=len(gt_imgs_path))):
        
        gt_id = gt_img_path.split('/')[-1].split('.')[0]
        lq_id = lq_img_path.split('/')[-1].split('.')[0]
        assert gt_id == lq_id, f"gt_img_path: {gt_img_path}, lq_img_path: {lq_img_path} do not match"
        
        gt_img = Image.open(gt_img_path)     # size: 512
        lq_img = Image.open(lq_img_path)     # size: 128
        
        # 记录原始图像尺寸
        original_gt_size = gt_img.size[::-1]  # PIL的size是(width, height)，需要转换为(height, width)
        original_lq_size = lq_img.size[::-1]

        # 分割成128x128的小块，重叠16像素
        gt_patches = split_image_with_overlap(gt_img, patch_size=128, overlap=16)
        lq_patches = split_image_with_overlap(lq_img, patch_size=128, overlap=16)

        restore_imgs = []
        lines = []
        # 处理每个patch
        for gt_patch, lq_patch in zip(gt_patches, lq_patches):
            val_gt = preprocess_gt(gt_patch).unsqueeze(0).to(device)  # 1 3 512 512
            val_lq = preprocess_lq(lq_patch).unsqueeze(0).to(device)  # 1 3 512 512
            val_bs, _, val_H, val_W = val_gt.shape
            
            val_prompt = [""]
                
            with torch.no_grad():
                val_clean = models['swinir'](val_lq)   
                val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

                M=1
                pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
                
                models['testr'].test_score_threshold = 0.5   
                ts_model = models['testr']

                # sampling
                val_z, val_ts_results = sampler.val_sample(    
                    model=models['cldm'],
                    device=device,
                    steps=50,
                    x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),   # manual shape adjustment
                    cond=val_cond,
                    uncond=None,
                    cfg_scale=1.0,
                    x_T = pure_noise,
                    progress=accelerator.is_main_process,
                    cfg=cfg, 
                    pure_cldm=pure_cldm,
                    ts_model = ts_model,
                    val_prompt=val_prompt
                )
                
                # log val prompts
                val_prompt = val_prompt[0]
                
                lines.append(f"** using OCR prompt w/ {cfg.exp_args.prompt_style}style **\n")
                # Format prompt
                lines.append("initial input prompt:\n")
                width = 80
                for i in range(0, len(val_prompt), width):
                    lines.append(val_prompt[i:i+width] + "\n")
                lines.append("\n")
                
                # Add prediction results
                for ts_result in val_ts_results:
                    timestep = ts_result['timestep']
                    pred_texts = ', '.join(ts_result['pred_texts'])
                    lines.append(f"timestep: {timestep:<4} /  pred_texts: {pred_texts}\n")
                
                
                
                restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)   # 1 3 512 512
                restore_imgs.append(restored_img)
            
        # 合并所有处理后的patches
        # 由于GT图像是512x512，LQ图像是128x128，我们需要根据实际情况选择原始尺寸
        # 这里假设我们要恢复到GT图像的尺寸
        final_restored_image = merge_patches_with_overlap(restore_imgs, original_gt_size, patch_size=512, overlap=64)
        
        # 现在final_restored_image是合并后的完整图像
        # 你可以继续进行后续的处理和保存
        
        
        # Now convert the list of strings to image
        img_of_pred_text = text_to_image(lines)
            
            
            # save sampled images   
        if cfg.log_args.log_tool is None:
            img_save_path = f'{cfg.exp_args.save_val_img_dir}'
            os.makedirs(img_save_path, exist_ok=True)
            restored_img_pil = TF.to_pil_image(final_restored_image.squeeze().cpu())
            restored_img_pil.save(f'{img_save_path}/restored_{gt_id}.png')
            img_of_pred_text.save(f'{img_save_path}/pred_texts_{gt_id}.png')
        
        
        # log total psnr, ssim, lpips for val
        # tot_val_psnr.append(torch.mean(metric_psnr(final_restored_image, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        # tot_val_ssim.append(torch.mean(metric_ssim(final_restored_image, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        # tot_val_lpips.append(torch.mean(metric_lpips(final_restored_image, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        # tot_val_dists.append(torch.mean(metric_dists(final_restored_image, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        # tot_val_niqe.append(torch.mean(metric_niqe(final_restored_image, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        # tot_val_musiq.append(torch.mean(metric_musiq(final_restored_image, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        # tot_val_maniqa.append(torch.mean(metric_maniqa(final_restored_image, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        # tot_val_clipiqa.append(torch.mean(metric_clipiqa(final_restored_image, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        
        # log sampling val imgs to wandb
        if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':

            # # log sampling val metrics 
            # wandb.log({f'sampling_val_METRIC/val_psnr': torch.mean(metric_psnr(
            #                                                                 final_restored_image, 
            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
            #         f'sampling_val_METRIC/val_ssim': torch.mean(metric_ssim(
            #                                                                 final_restored_image, 
            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
            #         f'sampling_val_METRIC/val_lpips': torch.mean(metric_lpips(
            #                                                                 final_restored_image, 
            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
            #         f'sampling_val_METRIC/val_dists': torch.mean(metric_dists(
            #                                                                 final_restored_image, 
            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
            #         f'sampling_val_METRIC/val_niqe': torch.mean(metric_niqe(
            #                                                                 final_restored_image, 
            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
            #         f'sampling_val_METRIC/val_musiq': torch.mean(metric_musiq(
            #                                                                 final_restored_image, 
            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
            #         f'sampling_val_METRIC/val_maniqa': torch.mean(metric_maniqa(
            #                                                                 final_restored_image, 
            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
            #         f'sampling_val_METRIC/val_clipiqa': torch.mean(metric_clipiqa(
            #                                                                 final_restored_image, 
            #                                                                 torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
            #         })
            
            # log sampling val images 
            wandb.log({ f'sampling_val_FINAL_VIS/{gt_id}_val_gt': wandb.Image((val_gt + 1) / 2, caption=f'gt_img'),
                        f'sampling_val_FINAL_VIS/{gt_id}_val_lq': wandb.Image(val_lq, caption=f'lq_img'),
                        f'sampling_val_FINAL_VIS/{gt_id}_val_cleaned': wandb.Image(val_clean, caption=f'cleaned_img'),
                        f'sampling_val_FINAL_VIS/{gt_id}_val_sampled': wandb.Image(torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), caption=f'sampled_img'),
                        f'sampling_val_FINAL_VIS/{gt_id}_val_prompts': wandb.Image(img_of_pred_text, caption='prompts used for sampling'),
                    })
            wandb.log({f'sampling_val_FINAL_VIS/{gt_id}_val_all': wandb.Image(torch.concat([val_lq, val_clean, torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, 0, 1), val_gt], dim=2), caption='lq_clean_sample,gt')})

        
    # average using numpy
    # tot_val_psnr = np.array(tot_val_psnr).mean()
    # tot_val_ssim = np.array(tot_val_ssim).mean()
    # tot_val_lpips = np.array(tot_val_lpips).mean()
    # tot_val_dists = np.array(tot_val_dists).mean()
    # tot_val_niqe = np.array(tot_val_niqe).mean()
    # tot_val_musiq = np.array(tot_val_musiq).mean()
    # tot_val_maniqa = np.array(tot_val_maniqa).mean()
    # tot_val_clipiqa = np.array(tot_val_clipiqa).mean()


    # # log total val metrics 
    # if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
    #     wandb.log({
    #         f'sampling_val_METRIC/tot_val_psnr': tot_val_psnr,
    #         f'sampling_val_METRIC/tot_val_ssim': tot_val_ssim,
    #         f'sampling_val_METRIC/tot_val_lpips': tot_val_lpips,
    #         f'sampling_val_METRIC/tot_val_dists': tot_val_dists,
    #         f'sampling_val_METRIC/tot_val_niqe': tot_val_niqe,
    #         f'sampling_val_METRIC/tot_val_musiq': tot_val_musiq,
    #         f'sampling_val_METRIC/tot_val_maniqa': tot_val_maniqa,
    #         f'sampling_val_METRIC/tot_val_clipiqa': tot_val_clipiqa,
    #     })
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)