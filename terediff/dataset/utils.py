import random
import math
import os
import json 
import numpy as np
from PIL import Image
import cv2
import torch
from .diffjpeg import DiffJPEG
from torch.nn import functional as F

# 导入中文字符集
from .chinese_vocab import CTLABELS, CHAR_TO_IDX, IDX_TO_CHAR, VOCAB_SIZE


def decode(idxs):
    """解码索引序列为文本"""
    s = ''
    for idx in idxs:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        if idx < len(CTLABELS):
            s += CTLABELS[idx]
        else:
            # 遇到未知索引就停止
            break
    return s


def encode(word):
    """编码文本为索引序列"""
    s = []
    max_word_len = 25  # 可能需要根据中文文本长度调整
    
    for i in range(max_word_len):
        if i < len(word):
            char = word[i]
            if char in CHAR_TO_IDX:
                s.append(CHAR_TO_IDX[char])
            else:
                # 未知字符用特殊标记
                s.append(VOCAB_SIZE)  # 或者跳过这个字符
        else:
            # 填充
            s.append(VOCAB_SIZE)  # 使用vocab_size作为padding
    return s


def is_valid_char(char):
    """检查字符是否在支持的字符集中"""
    return char in CHAR_TO_IDX


def load_file_list(file_list_path: str, data_args=None):

    mode = data_args['mode']
    datasets = data_args['datasets']
    ann_path = data_args['ann_path']
    model_H, model_W = data_args['model_img_size']

    files = []
    for dataset in datasets:

        if dataset == 'sam_cleaned_100k':
            
            # 加载JSON数据
            json_path = ann_path 
            with open(json_path, 'r', encoding='utf-8') as f:  # 添加utf-8编码
                json_data = json.load(f)
                json_data = sorted(json_data.items())
            

            # 训练/验证分割
            split_index = int(len(json_data) * 10 / 11)
            if mode == 'TRAIN':
                json_data = dict(json_data[:split_index])
            elif mode == 'VAL':
                json_data = dict(json_data[split_index:])


            # image path 
            imgs_path = f'{file_list_path}/images'
            imgs = sorted(os.listdir(imgs_path))


            for img in imgs:
                gt_path = f'{imgs_path}/{img}'

                img_id = img.split('.')[0]
                if img_id in json_data.keys():
                    img_ann = json_data[img_id]['0']['text_instances']
                else:
                    continue


                boxes=[]
                texts=[]
                text_encs=[]
                polys=[]

                for ann in img_ann:

                    # 处理文本 - 修改的关键部分
                    text = ann['text']
                    
                    # 检查文本中的字符是否都在支持的字符集中
                    valid_char_count = 0
                    for char in text:
                        if is_valid_char(char):
                            valid_char_count += 1
                    
                    # 只接受完全支持的文本，并且长度合理
                    if valid_char_count == len(text) and len(text) <= 25 and len(text) > 0:
                        texts.append(text)
                        try:
                            encoded_text = encode(text)
                            text_encs.append(encoded_text)
                            # 验证编码解码的一致性
                            decoded_text = decode(encoded_text)
                            if decoded_text.strip() != text.strip():
                                print(f"Warning: Encoding/decoding mismatch for '{text}' -> '{decoded_text}'")
                                continue
                        except Exception as e:
                            print(f"Error encoding text '{text}': {e}")
                            continue
                    else:
                        # 跳过包含不支持字符的文本
                        continue


                    # 处理边界框
                    box_xyxy = ann['bbox']
                    x1,y1,x2,y2 = box_xyxy
                    box_xyxy_scaled = list(map(lambda x: x/model_H, box_xyxy))  # scale box coord to [0,1]
                    x1,y1,x2,y2 = box_xyxy_scaled 
                    box_cxcywh = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]   # xyxy -> cxcywh
                    # box format
                    processed_box = box_cxcywh
                    processed_box = list(map(lambda x: round(x,4), processed_box))
                    boxes.append(processed_box)


                    # 处理多边形
                    poly = np.array(ann['polygon']).astype(np.int32)    # 16 2
                    # scale poly
                    poly_scaled = poly / np.array([model_W, model_H])
                    polys.append(poly_scaled)


                    # # VISUALIZE FOR DEBUGGING
                    # img0 = cv2.imread(gt_path)  # 512 512 3
                    # x,y,w,h = box_xywh
                    # cv2.rectangle(img0_box, (x,y), (x+w, y+h), (0,255,0), 2)
                    # cv2.polylines(img0_poly, [poly], True, (0,255,0), 2)
                    # cv2.putText(img0_box, text, (poly[0][0], poly[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    # cv2.putText(img0_poly, text, (poly[0][0], poly[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    # cv2.imwrite('./img0_box.jpg', img0_box)
                    # cv2.imwrite('./img0_poly.jpg', img0_poly)

                assert len(boxes) == len(texts) == len(text_encs) == len(polys), f"Check loader!"

                # 如果过滤后没有有效的文本框，跳过这张图片
                if len(boxes) == 0 or len(polys) == 0:
                    continue
            
                # 生成描述文本
                caption = [f'"{txt}"' for txt in texts]
                prompt = f"A realistic scene where the texts {', '.join(caption)} appear clearly on signs, boards, buildings, or other objects."

                files.append({"image_path": gt_path, 
                              "prompt": prompt, 
                              "text": texts, 
                              "bbox": boxes,
                              'poly': polys,
                              'text_enc': text_encs, 
                              "img_name": img_id})     
    

    if mode=='VAL':
        files = random.sample(files, min(6, len(files)))

    return files


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/data/transforms.py
def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/img_process_util.py
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode="reflect")
    else:
        raise ValueError("Wrong kernel size")

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        # img: torch.Tensor
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer("kernel", kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img
