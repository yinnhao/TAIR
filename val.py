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


def make_divisible_by_window_size(size, window_size=8):
    """
    确保尺寸能被窗口大小整除
    """
    return ((size + window_size - 1) // window_size) * window_size


def crop_image_with_overlap(image, crop_size, overlap, window_size=8):
    """
    将图像裁剪成带有重叠区域的小块，确保尺寸能被窗口大小整除
    
    Args:
        image: PIL Image或torch.Tensor
        crop_size: 裁剪尺寸 (height, width) - 这是LQ图像的裁剪尺寸
        overlap: 重叠像素数
        window_size: 窗口大小，默认8
    
    Returns:
        crops: 裁剪后的图像块列表
        positions: 每个块在原图中的位置 [(x, y), ...]
        original_size: 原始图像尺寸
        actual_crop_size: 实际裁剪尺寸
    """
    if isinstance(image, torch.Tensor):
        _, _, h, w = image.shape
    else:
        w, h = image.size
    
    # 确保裁剪尺寸能被窗口大小整除
    crop_h = make_divisible_by_window_size(crop_size[0], window_size)
    crop_w = make_divisible_by_window_size(crop_size[1], window_size)
    
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
    
    return crops, positions, (w, h), (crop_w, crop_h)


def merge_crops_with_overlap(crops, positions, original_size, actual_crop_size, overlap, scale_factor=4):
    """
    将带有重叠区域的图像块合并回原始尺寸
    
    Args:
        crops: 处理后的图像块列表 (torch.Tensor) - 这些是超分后的图像块
        positions: 每个块在原图中的位置 (基于LQ图像的位置)
        original_size: 原始LQ图像尺寸 (width, height)
        actual_crop_size: 实际LQ裁剪尺寸 (width, height)
        overlap: 重叠像素数 (基于LQ图像的重叠)
        scale_factor: 超分倍数，默认4
    
    Returns:
        merged_image: 合并后的图像 (torch.Tensor)
    """
    lq_w, lq_h = original_size
    lq_crop_w, lq_crop_h = actual_crop_size
    
    # 计算超分后的尺寸
    sr_w = lq_w * scale_factor
    sr_h = lq_h * scale_factor
    sr_crop_w = lq_crop_w * scale_factor
    sr_crop_h = lq_crop_h * scale_factor
    sr_overlap = overlap * scale_factor
    
    device = crops[0].device
    
    # 初始化输出图像和权重图 (超分后的尺寸)
    merged = torch.zeros((1, 3, sr_h, sr_w), device=device)
    weights = torch.zeros((1, 1, sr_h, sr_w), device=device)
    
    # 创建权重模板（中心权重高，边缘权重低）
    weight_template = torch.ones((1, 1, sr_crop_h, sr_crop_w), device=device)
    if sr_overlap > 0:
        # 使用线性渐变作为权重
        fade_in = torch.linspace(0, 1, sr_overlap, device=device)
        fade_out = torch.linspace(1, 0, sr_overlap, device=device)
        
        # 上边界
        weight_template[:, :, :sr_overlap, :] *= fade_in.view(-1, 1)
        # 下边界
        weight_template[:, :, -sr_overlap:, :] *= fade_out.view(-1, 1)
        # 左边界
        weight_template[:, :, :, :sr_overlap] *= fade_in.view(1, -1)
        # 右边界
        weight_template[:, :, :, -sr_overlap:] *= fade_out.view(1, -1)
    
    # 合并每个裁剪块
    for crop, (lq_x, lq_y) in zip(crops, positions):
        # 将LQ位置转换为SR位置
        sr_x = lq_x * scale_factor
        sr_y = lq_y * scale_factor
        
        merged[:, :, sr_y:sr_y+sr_crop_h, sr_x:sr_x+sr_crop_w] += crop * weight_template
        weights[:, :, sr_y:sr_y+sr_crop_h, sr_x:sr_x+sr_crop_w] += weight_template
    
    # 归一化
    merged = merged / (weights + 1e-8)
    
    return merged


def pad_image_to_window_size(image, window_size=8):
    """
    将图像填充到能被窗口大小整除的尺寸
    """
    if isinstance(image, torch.Tensor):
        _, _, h, w = image.shape
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, (0, 0, pad_w, pad_h), padding_mode='reflect')
    else:
        w, h = image.size
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            # 对PIL图像进行填充
            new_w = w + pad_w
            new_h = h + pad_h
            padded_image = Image.new(image.mode, (new_w, new_h))
            padded_image.paste(image, (0, 0))
            # 使用反射填充剩余区域
            if pad_w > 0:
                # 右边填充
                right_strip = image.crop((w-pad_w, 0, w, h))
                right_strip = right_strip.transpose(Image.FLIP_LEFT_RIGHT)
                padded_image.paste(right_strip, (w, 0))
            if pad_h > 0:
                # 下边填充
                bottom_strip = image.crop((0, h-pad_h, w, h))
                bottom_strip = bottom_strip.transpose(Image.FLIP_TOP_BOTTOM)
                padded_image.paste(bottom_strip, (0, h))
            if pad_w > 0 and pad_h > 0:
                # 右下角填充
                corner = image.crop((w-pad_w, h-pad_h, w, h))
                corner = corner.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
                padded_image.paste(corner, (w, h))
            image = padded_image
    
    return image


def process_image_with_crops(lq_image_path, gt_image_path, models, pure_cldm, sampler, device, gen, cfg, 
                           lq_crop_size=(128, 128), overlap=16, preprocess_lq=None, preprocess_gt=None, window_size=8):
    """
    使用裁剪方式处理大图像
    
    Args:
        lq_image_path: LQ图像路径
        gt_image_path: GT图像路径  
        models: 模型字典
        pure_cldm: ControlLDM模型
        sampler: 采样器
        device: 设备
        gen: 随机数生成器
        cfg: 配置
        lq_crop_size: LQ图像裁剪尺寸 (height, width)
        overlap: 重叠区域大小 (基于LQ图像)
        preprocess_lq: LQ预处理变换
        preprocess_gt: GT预处理变换
        window_size: 窗口大小
    
    Returns:
        restored_img: 恢复的图像 (超分后的尺寸)
        val_gt: GT图像
        val_lq: LQ图像 (调整到超分后的尺寸用于显示)
        val_clean: 清理后的图像
        all_ts_results: 所有时间步结果
    """
    # 加载图像
    lq_img = Image.open(lq_image_path)    # 128x128 (原始LQ尺寸)
    gt_img = Image.open(gt_image_path)    # 512x512 (原始GT尺寸)
    
    original_lq_size = lq_img.size
    original_gt_size = gt_img.size
    
    # 确保LQ图像尺寸能被窗口大小整除
    lq_img = pad_image_to_window_size(lq_img, window_size)
    
    # 如果LQ图像小于裁剪尺寸，直接处理
    if lq_img.size[0] <= lq_crop_size[1] and lq_img.size[1] <= lq_crop_size[0]:
        # 处理LQ图像
        if preprocess_lq:
            processed_lq = preprocess_lq(lq_img).unsqueeze(0).to(device)
        else:
            processed_lq = T.ToTensor()(lq_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            val_clean = models['swinir'](processed_lq)
            val_cond = pure_cldm.prepare_condition(val_clean, [""])
            
            # 确保噪声尺寸正确 (基于LQ图像尺寸)
            noise_h = make_divisible_by_window_size(lq_crop_size[0], 8) // 8
            noise_w = make_divisible_by_window_size(lq_crop_size[1], 8) // 8
            pure_noise = torch.randn((1, 4, noise_h, noise_w), generator=gen, device=device, dtype=torch.float32)
            
            models['testr'].test_score_threshold = 0.5
            ts_model = models['testr']
            
            val_z, val_ts_results = sampler.val_sample(
                model=models['cldm'],
                device=device,
                steps=50,
                x_size=(1, 4, noise_h, noise_w),
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
            
            # 裁剪回原始GT尺寸
            if original_gt_size != (lq_img.size[0] * 4, lq_img.size[1] * 4):
                restored_img = restored_img[:, :, :original_gt_size[1], :original_gt_size[0]]
        
        # 处理GT图像
        if preprocess_gt:
            val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)
        else:
            val_gt = T.ToTensor()(gt_img).unsqueeze(0).to(device)
            val_gt = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(val_gt)
        
        # 调整GT尺寸以匹配restored图像
        if val_gt.shape != restored_img.shape:
            val_gt = T.Resize(size=(restored_img.shape[2], restored_img.shape[3]), 
                             interpolation=T.InterpolationMode.BICUBIC)(val_gt)
        
        # 为显示创建放大的LQ图像
        val_lq = T.ToTensor()(lq_img).unsqueeze(0).to(device)
        val_lq = T.Resize(size=(restored_img.shape[2], restored_img.shape[3]), 
                         interpolation=T.InterpolationMode.BICUBIC)(val_lq)
        
        return restored_img, val_gt, val_lq, val_clean, [val_ts_results]
    
    # 裁剪LQ图像
    lq_crops, lq_positions, padded_lq_size, actual_lq_crop_size = crop_image_with_overlap(lq_img, lq_crop_size, overlap, window_size)
    
    restored_crops = []
    all_ts_results = []
    
    print(f"Processing {len(lq_crops)} crops for image {lq_image_path}")
    
    for i, lq_crop in enumerate(tqdm(lq_crops, desc="Processing crops")):
        # 预处理LQ裁剪块
        if preprocess_lq:
            processed_lq_crop = preprocess_lq(lq_crop).unsqueeze(0).to(device)
        else:
            processed_lq_crop = T.ToTensor()(lq_crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # 清理图像
            val_clean = models['swinir'](processed_lq_crop)
            val_cond = pure_cldm.prepare_condition(val_clean, [""])
            
            # 生成噪声 (基于LQ裁剪尺寸)
            noise_h = actual_lq_crop_size[1] // 8
            noise_w = actual_lq_crop_size[0] // 8
            pure_noise = torch.randn((1, 4, noise_h, noise_w), 
                                   generator=gen, device=device, dtype=torch.float32)
            
            models['testr'].test_score_threshold = 0.5
            ts_model = models['testr']
            
            # 采样
            val_z, val_ts_results = sampler.val_sample(
                model=models['cldm'],
                device=device,
                steps=50,
                x_size=(1, 4, noise_h, noise_w),
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
            
            # 解码 (得到4倍超分后的图像块)
            restored_crop = torch.clamp((pure_cldm.vae_decode(val_z) + 1) / 2, min=0, max=1)
            restored_crops.append(restored_crop)
            all_ts_results.append(val_ts_results)
    
    # 合并裁剪块 (输出为4倍超分后的尺寸)
    merged_image = merge_crops_with_overlap(restored_crops, lq_positions, padded_lq_size, actual_lq_crop_size, overlap, scale_factor=4)
    
    # 裁剪回原始GT尺寸
    if original_gt_size != (padded_lq_size[0] * 4, padded_lq_size[1] * 4):
        merged_image = merged_image[:, :, :original_gt_size[1], :original_gt_size[0]]
    
    # 处理GT图像
    if preprocess_gt:
        val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)
    else:
        val_gt = T.ToTensor()(gt_img).unsqueeze(0).to(device)
        val_gt = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(val_gt)
    
    # 调整GT尺寸以匹配restored图像
    if val_gt.shape != merged_image.shape:
        val_gt = T.Resize(size=(merged_image.shape[2], merged_image.shape[3]), 
                         interpolation=T.InterpolationMode.BICUBIC)(val_gt)
    
    # 为显示创建放大的LQ图像
    val_lq = T.ToTensor()(lq_img).unsqueeze(0).to(device)
    val_lq = T.Resize(size=(merged_image.shape[2], merged_image.shape[3]), 
                     interpolation=T.InterpolationMode.BICUBIC)(val_lq)
    
    # 创建清理后的图像用于显示
    val_clean = models['swinir'](val_lq)
    
    return merged_image, val_gt, val_lq, val_clean, all_ts_results


def main(args):
    # set accelerator, seed, device, config
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, kwargs_handlers=[kwargs])
    set_seed(25, device_specific=False)
    device = accelerator.device
    gen = torch.Generator(device)
    cfg = OmegaConf.load(args.config)
    
    # 添加裁剪配置
    lq_crop_size = getattr(cfg.dataset, 'lq_crop_size', (128, 128))  # LQ裁剪尺寸
    overlap = getattr(cfg.dataset, 'overlap', 16)  # 基于LQ图像的重叠
    enable_crop = getattr(cfg.dataset, 'enable_crop', False)
    window_size = getattr(cfg.dataset, 'window_size', 8)  # SwinIR的窗口大小
    
    # 确保LQ裁剪尺寸能被窗口大小整除
    lq_crop_size = (make_divisible_by_window_size(lq_crop_size[0], window_size), 
                   make_divisible_by_window_size(lq_crop_size[1], window_size))
    
    # GT裁剪尺寸是LQ的4倍
    gt_crop_size = (lq_crop_size[0] * 4, lq_crop_size[1] * 4)
    
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

    # For val_gt (range [-1, 1]) - GT图像预处理
    preprocess_gt = T.Compose([
        T.Resize(size=gt_crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # For val_lq (range [0, 1]) - LQ图像预处理
    preprocess_lq = T.Compose([
        T.Resize(size=lq_crop_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor()
    ])
    
    for val_batch_idx, (gt_img_path, lq_img_path) in enumerate(tqdm(zip(gt_imgs_path, lq_imgs_path), desc='val', total=len(gt_imgs_path))):
        
        gt_id = gt_img_path.split('/')[-1].split('.')[0]
        lq_id = lq_img_path.split('/')[-1].split('.')[0]
        assert gt_id == lq_id, f"gt_img_path: {gt_img_path}, lq_img_path: {lq_img_path} do not match"
        
        if enable_crop:
            # 使用裁剪方式处理
            restored_img, val_gt, val_lq, val_clean, all_ts_results = process_image_with_crops(
                lq_img_path, gt_img_path, models, pure_cldm, sampler, device, gen, cfg,
                lq_crop_size=lq_crop_size, overlap=overlap, 
                preprocess_lq=preprocess_lq, preprocess_gt=preprocess_gt, window_size=window_size
            )
            
            # 合并所有时间步结果用于显示
            val_ts_results = []
            for ts_results in all_ts_results:
                val_ts_results.extend(ts_results)
            
        else:
            # 原始处理方式
            gt_img = Image.open(gt_img_path)     # size: 512
            lq_img = Image.open(lq_img_path)     # size: 128
            
            val_gt = preprocess_gt(gt_img).unsqueeze(0).to(device)  # 1 3 512 512
            val_lq = preprocess_lq(lq_img).unsqueeze(0).to(device)  # 1 3 128 128
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
                
                # 为显示调整LQ图像尺寸
                val_lq = T.Resize(size=(val_H, val_W), interpolation=T.InterpolationMode.BICUBIC)(val_lq)
        
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
