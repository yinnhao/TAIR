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


def crop_image_with_overlap(image, crop_size, overlap):
    """
    将图像裁剪成带有重叠区域的小块
    
    Args:
        image: PIL Image或torch.Tensor
        crop_size: 裁剪尺寸 (height, width)
        overlap: 重叠像素数
    
    Returns:
        crops: 裁剪后的图像块列表
        positions: 每个块在原图中的位置 [(x, y), ...]
        original_size: 原始图像尺寸
    """
    if isinstance(image, torch.Tensor):
        _, _, h, w = image.shape
    else:
        w, h = image.size
    
    crop_h, crop_w = crop_size
    stride_h = crop_h - overlap
    stride_w = crop_w - overlap
    
    crops = []
    positions = []
    
    for y in range(0, h - crop_h + 1, stride_h):
        for x in range(0, w - crop_w + 1, stride_w):
            if isinstance(image, torch.Tensor):
                crop = image[:, :, y:y+crop_h, x:x+crop_w]
            else:
                crop = image.crop((x, y, x+crop_w, y+crop_h))
            crops.append(crop)
            positions.append((x, y))
    
    # 处理边界情况
    # 右边界
    if (w - crop_w) % stride_w != 0:
        x = w - crop_w
        for y in range(0, h - crop_h + 1, stride_h):
            if isinstance(image, torch.Tensor):
                crop = image[:, :, y:y+crop_h, x:x+crop_w]
            else:
                crop = image.crop((x, y, x+crop_w, y+crop_h))
            crops.append(crop)
            positions.append((x, y))
    
    # 下边界
    if (h - crop_h) % stride_h != 0:
        y = h - crop_h
        for x in range(0, w - crop_w + 1, stride_w):
            if isinstance(image, torch.Tensor):
                crop = image[:, :, y:y+crop_h, x:x+crop_w]
            else:
                crop = image.crop((x, y, x+crop_w, y+crop_h))
            crops.append(crop)
            positions.append((x, y))
    
    # 右下角
    if (w - crop_w) % stride_w != 0 and (h - crop_h) % stride_h != 0:
        x, y = w - crop_w, h - crop_h
        if isinstance(image, torch.Tensor):
            crop = image[:, :, y:y+crop_h, x:x+crop_w]
        else:
            crop = image.crop((x, y, x+crop_w, y+crop_h))
        crops.append(crop)
        positions.append((x, y))
    
    return crops, positions, (w, h)


def merge_crops_with_overlap(crops, positions, original_size, crop_size, overlap):
    """
    将带有重叠区域的图像块合并回原始尺寸
    
    Args:
        crops: 处理后的图像块列表 (torch.Tensor)
        positions: 每个块在原图中的位置
        original_size: 原始图像尺寸 (width, height)
        crop_size: 裁剪尺寸 (height, width)
        overlap: 重叠像素数
    
    Returns:
        merged_image: 合并后的图像 (torch.Tensor)
    """
    w, h = original_size
    crop_h, crop_w = crop_size
    device = crops[0].device
    
    # 初始化输出图像和权重图
    merged = torch.zeros((1, 3, h, w), device=device)
    weights = torch.zeros((1, 1, h, w), device=device)
    
    # 创建权重模板（中心权重高，边缘权重低）
    weight_template = torch.ones((1, 1, crop_h, crop_w), device=device)
    if overlap > 0:
        # 使用线性渐变作为权重
        fade_in = torch.linspace(0, 1, overlap, device=device)
        fade_out = torch.linspace(1, 0, overlap, device=device)
        
        # 上边界
        weight_template[:, :, :overlap, :] *= fade_in.view(-1, 1)
        # 下边界
        weight_template[:, :, -overlap:, :] *= fade_out.view(-1, 1)
        # 左边界
        weight_template[:, :, :, :overlap] *= fade_in.view(1, -1)
        # 右边界
        weight_template[:, :, :, -overlap:] *= fade_out.view(1, -1)
    
    # 合并每个裁剪块
    for crop, (x, y) in zip(crops, positions):
        merged[:, :, y:y+crop_h, x:x+crop_w] += crop * weight_template
        weights[:, :, y:y+crop_h, x:x+crop_w] += weight_template
    
    # 归一化
    merged = merged / (weights + 1e-8)
    
    return merged


def process_image_with_crops(image_path, models, pure_cldm, sampler, device, gen, cfg, 
                           crop_size=(512, 512), overlap=64, preprocess_transforms=None):
    """
    使用裁剪方式处理大图像
    
    Args:
        image_path: 图像路径
        models: 模型字典
        pure_cldm: ControlLDM模型
        sampler: 采样器
        device: 设备
        gen: 随机数生成器
        cfg: 配置
        crop_size: 裁剪尺寸
        overlap: 重叠区域大小
        preprocess_transforms: 预处理变换
    
    Returns:
        restored_img: 恢复的图像
        all_ts_results: 所有时间步结果
    """
    # 加载图像
    img = Image.open(image_path)
    original_size = img.size
    
    # 如果图像小于裁剪尺寸，直接处理
    if img.size[0] <= crop_size[1] and img.size[1] <= crop_size[0]:
        if preprocess_transforms:
            processed_img = preprocess_transforms(img).unsqueeze(0).to(device)
        else:
            processed_img = T.ToTensor()(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            val_clean = models['swinir'](processed_img)
            val_cond = pure_cldm.prepare_condition(val_clean, [""])
            
            pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
            models['testr'].test_score_threshold = 0.5
            ts_model = models['testr']
            
            val_z, val_ts_results = sampler.val_sample(
                model=models['cldm'],
                device=device,
                steps=50,
                x_size=(1, 4, crop_size[0]//8, crop_size[1]//8),
                cond=val_cond,
                uncond=None,
                cfg_scale=1.0,
                x_T=pure_noise,
                progress=False,
                cfg=cfg,
                pure_cldm=pure_cldm,
                ts_model=ts_model,
                val_prompt=[""]
            )
            
            restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)
        
        return restored_img, [val_ts_results]
    
    # 裁剪图像
    crops, positions, original_size = crop_image_with_overlap(img, crop_size, overlap)
    
    restored_crops = []
    all_ts_results = []
    
    print(f"Processing {len(crops)} crops for image {image_path}")
    
    for i, crop in enumerate(tqdm(crops, desc="Processing crops")):
        # 预处理裁剪块
        if preprocess_transforms:
            processed_crop = preprocess_transforms(crop).unsqueeze(0).to(device)
        else:
            processed_crop = T.ToTensor()(crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 清理图像
            val_clean = models['swinir'](processed_crop)
            val_cond = pure_cldm.prepare_condition(val_clean, [""])
            
            # 生成噪声
            pure_noise = torch.randn((1, 4, crop_size[0]//8, crop_size[1]//8), 
                                   generator=gen, device=device, dtype=torch.float32)
            
            models['testr'].test_score_threshold = 0.5
            ts_model = models['testr']
            
            # 采样
            val_z, val_ts_results = sampler.val_sample(
                model=models['cldm'],
                device=device,
                steps=50,
                x_size=(1, 4, crop_size[0]//8, crop_size[1]//8),
                cond=val_cond,
                uncond=None,
                cfg_scale=1.0,
                x_T=pure_noise,
                progress=False,
                cfg=cfg,
                pure_cldm=pure_cldm,
                ts_model=ts_model,
                val_prompt=[""]
            )
            
            # 解码
            restored_crop = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)
            restored_crops.append(restored_crop)
            all_ts_results.append(val_ts_results)
    
    # 合并裁剪块
    merged_image = merge_crops_with_overlap(restored_crops, positions, original_size, crop_size, overlap)
    
    return merged_image, all_ts_results


def main(args):
    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(25, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    cfg = OmegaConf.load(args.config)
    
    # 添加裁剪配置
    crop_size = getattr(cfg.dataset, 'crop_size', (512, 512))
    overlap = getattr(cfg.dataset, 'overlap', 64)
    enable_crop = getattr(cfg.dataset, 'enable_crop', False)
    
    # setup logging tool
    if cfg.log_args.log_tool == 'wandb':
        wandb.login(key=cfg.log_args.wandb_key)
        wandb.init(project=cfg.log_args.wandb_proj_name, 
                name='VAL_terediff_stage3_DEMO_CROP' if enable_crop else 'VAL_terediff_stage3_DEMO',
                config=argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))
        )

    # load demo images from demo_imgs/ folder
    gt_imgs_path = sorted([f"{cfg.dataset.gt_img_path}/{img}" for img in os.listdir(cfg.dataset.gt_img_path) if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")])
    lq_imgs_path = sorted([f"{cfg.dataset.lq_img_path}/{img}" for img in os.listdir(cfg.dataset.lq_img_path) if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")])

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
        T.Resize(size=crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # For val_lq (range [0, 1])
    preprocess_lq = T.Compose([
        T.Resize(size=crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    for val_batch_idx, (gt_img_path, lq_img_path) in enumerate(tqdm(zip(gt_imgs_path, lq_imgs_path), desc='val', total=len(gt_imgs_path))):
        
        gt_id = gt_img_path.split('/')[-1].split('.')[0]
        lq_id = lq_img_path.split('/')[-1].split('.')[0]
        assert gt_id == lq_id, f"gt_img_path: {gt_img_path}, lq_img_path: {lq_img_path} do not match"
        
        if enable_crop:
            # 使用裁剪方式处理
            restored_img, all_ts_results = process_image_with_crops(
                lq_img_path, models, pure_cldm, sampler, device, gen, cfg,
                crop_size=crop_size, overlap=overlap, preprocess_transforms=preprocess_lq
            )
            
            # 处理GT图像以匹配尺寸
            gt_img = Image.open(gt_img_path)
            gt_tensor = T.ToTensor()(gt_img).unsqueeze(0).to(device)
            gt_tensor = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(gt_tensor)
            
            # 调整GT图像尺寸以匹配restored图像
            if gt_tensor.shape != restored_img.shape:
                gt_tensor = T.Resize(size=(restored_img.shape[2], restored_img.shape[3]), 
                                   interpolation=T.InterpolationMode.BICUBIC)(gt_tensor)
            
            val_gt = gt_tensor
            
            # 为了显示，也处理LQ图像
            lq_img = Image.open(lq_img_path)
            val_lq = T.ToTensor()(lq_img).unsqueeze(0).to(device)
            if val_lq.shape != restored_img.shape:
                val_lq = T.Resize(size=(restored_img.shape[2], restored_img.shape[3]), 
                                interpolation=T.InterpolationMode.BICUBIC)(val_lq)
            
            # 合并所有时间步结果用于显示
            val_ts_results = []
            for ts_results in all_ts_results:
                val_ts_results.extend(ts_results)
            
            # 创建清理后的图像用于显示
            val_clean = models['swinir'](val_lq)
            
        else:
            # 原始处理方式
            gt_img = Image.open(gt_img_path)
            lq_img = Image.open(lq_img_path)
            
            val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)
            val_lq = preprocess_lq(lq_img).unsqueeze(0).to(device)
            val_bs, _, val_H, val_W = val_gt.shape
            
            val_prompt = [""]
                
            with torch.no_grad():
                val_clean = models['swinir'](val_lq)   
                val_cond = pure_cldm.prepare_condition(val_clean, val_prompt)

                pure_noise = torch.randn((1, 4, 64, 64), generator=gen, device=device, dtype=torch.float32)
                
                models['testr'].test_score_threshold = 0.5   
                ts_model = models['testr']

                # sampling
                val_z, val_ts_results = sampler.val_sample(    
                    model=models['cldm'],
                    device=device,
                    steps=50,
                    x_size=(val_bs, 4, int(val_H/8), int(val_W/8)),
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
                
                restored_img = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)
        
        # log val prompts
        val_prompt = [""] if enable_crop else val_prompt[0]
        lines = []
        lines.append(f"** using OCR prompt w/ {cfg.exp_args.prompt_style}style **\n")
        lines.append("initial input prompt:\n")
        width = 80
        prompt_text = val_prompt if isinstance(val_prompt, str) else val_prompt[0]
        for i in range(0, len(prompt_text), width):
            lines.append(prompt_text[i:i+width] + "\n")
        lines.append("\n")
        
        # Add prediction results
        for ts_result in val_ts_results:
            timestep = ts_result['timestep']
            pred_texts = ', '.join(ts_result['pred_texts'])
            lines.append(f"timestep: {timestep:<4} /  pred_texts: {pred_texts}\n")
        
        # Now convert the list of strings to image
        img_of_pred_text = text_to_image(lines)
        
        # save sampled images   
        if cfg.log_args.log_tool is None:
            img_save_path = f'{cfg.exp_args.save_val_img_dir}'
            os.makedirs(img_save_path, exist_ok=True)
            restored_img_pil = TF.to_pil_image(restored_img.squeeze().cpu())
            restored_img_pil.save(f'{img_save_path}/restored_{gt_id}.png')
            img_of_pred_text.save(f'{img_save_path}/pred_texts_{gt_id}.png')
        
        # log total psnr, ssim, lpips for val
        tot_val_psnr.append(torch.mean(metric_psnr(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        tot_val_ssim.append(torch.mean(metric_ssim(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        tot_val_lpips.append(torch.mean(metric_lpips(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        tot_val_dists.append(torch.mean(metric_dists(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        tot_val_niqe.append(torch.mean(metric_niqe(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        tot_val_musiq.append(torch.mean(metric_musiq(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        tot_val_maniqa.append(torch.mean(metric_maniqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        tot_val_clipiqa.append(torch.mean(metric_clipiqa(restored_img, torch.clamp((val_gt + 1) / 2, min=0, max=1))).item())
        
        # log sampling val imgs to wandb
        if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':

            # log sampling val metrics 
            wandb.log({f'sampling_val_METRIC/val_psnr': torch.mean(metric_psnr(
                                                                            restored_img, 
                                                                            torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                    f'sampling_val_METRIC/val_ssim': torch.mean(metric_ssim(
                                                                            restored_img, 
                                                                            torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                    f'sampling_val_METRIC/val_lpips': torch.mean(metric_lpips(
                                                                            restored_img, 
                                                                            torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                    f'sampling_val_METRIC/val_dists': torch.mean(metric_dists(
                                                                            restored_img, 
                                                                            torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                    f'sampling_val_METRIC/val_niqe': torch.mean(metric_niqe(
                                                                            restored_img, 
                                                                            torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                    f'sampling_val_METRIC/val_musiq': torch.mean(metric_musiq(
                                                                            restored_img, 
                                                                            torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                    f'sampling_val_METRIC/val_maniqa': torch.mean(metric_maniqa(
                                                                            restored_img, 
                                                                            torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                    f'sampling_val_METRIC/val_clipiqa': torch.mean(metric_clipiqa(
                                                                            restored_img, 
                                                                            torch.clamp((val_gt + 1) / 2, min=0, max=1))).item(),
                    })
            
            # log sampling val images 
            wandb.log({ f'sampling_val_FINAL_VIS/{gt_id}_val_gt': wandb.Image((val_gt + 1) / 2, caption=f'gt_img'),
                        f'sampling_val_FINAL_VIS/{gt_id}_val_lq': wandb.Image(val_lq, caption=f'lq_img'),
                        f'sampling_val_FINAL_VIS/{gt_id}_val_cleaned': wandb.Image(val_clean, caption=f'cleaned_img'),
                        f'sampling_val_FINAL_VIS/{gt_id}_val_sampled': wandb.Image(restored_img, caption=f'sampled_img'),
                        f'sampling_val_FINAL_VIS/{gt_id}_val_prompts': wandb.Image(img_of_pred_text, caption='prompts used for sampling'),
                    })
            wandb.log({f'sampling_val_FINAL_VIS/{gt_id}_val_all': wandb.Image(torch.concat([val_lq, val_clean, restored_img, (val_gt + 1) / 2], dim=2), caption='lq_clean_sample_gt')})

    # average using numpy
    tot_val_psnr = np.array(tot_val_psnr).mean()
    tot_val_ssim = np.array(tot_val_ssim).mean()
    tot_val_lpips = np.array(tot_val_lpips).mean()
    tot_val_dists = np.array(tot_val_dists).mean()
    tot_val_niqe = np.array(tot_val_niqe).mean()
    tot_val_musiq = np.array(tot_val_musiq).mean()
    tot_val_maniqa = np.array(tot_val_maniqa).mean()
    tot_val_clipiqa = np.array(tot_val_clipiqa).mean()

    # log total val metrics 
    if accelerator.is_main_process and cfg.log_args.log_tool == 'wandb':
        wandb.log({
            f'sampling_val_METRIC/tot_val_psnr': tot_val_psnr,
            f'sampling_val_METRIC/tot_val_ssim': tot_val_ssim,
            f'sampling_val_METRIC/tot_val_lpips': tot_val_lpips,
            f'sampling_val_METRIC/tot_val_dists': tot_val_dists,
            f'sampling_val_METRIC/tot_val_niqe': tot_val_niqe,
            f'sampling_val_METRIC/tot_val_musiq': tot_val_musiq,
            f'sampling_val_METRIC/tot_val_maniqa': tot_val_maniqa,
            f'sampling_val_METRIC/tot_val_clipiqa': tot_val_clipiqa,
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--config_testr', type=str)
    args = parser.parse_args()
    main(args)
