import numpy as np  #t
import cv2
import argparse
import sys
import os
from t3_dataset import draw_glyph2
from PIL import Image, ImageFont

def render_glyph_image(
    mask: np.ndarray,
    text: str,
    width: int,
    height: int,
    font_path: str,
    font_size: int = 60,
    save_path: str = None
) -> np.ndarray:
    """
    从掩码中提取轮廓，并在指定尺寸内绘制glyph图像。
    可选地将图像保存到磁盘。

    参数：
        mask: numpy数组，二值图像，表示绘制区域
        text: 要绘制的文字内容
        width: 目标图像宽度
        height: 目标图像高度
        font_path: 字体路径
        font_size: 字体大小（默认60）
        save_path: 可选，若指定路径则保存图像为PNG

    返回：
        glyph_img: np.ndarray，绘制好的图像（float 类型，范围 0~1）
    """
    if mask.ndim != 2:
        raise ValueError("mask 应该是单通道二维图像")
    
    # 提取轮廓
    mask = mask.astype('uint8')
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("未找到有效轮廓")

    # 强制转换轮廓数据类型为 int32（OpenCV 要求）
    contour = contours[0].astype(np.float32)
    
    # 加载字体
    selffont = ImageFont.truetype(font_path, size=font_size)

    # 绘制 glyph 图像（float 类型，值在 0~1）
    glyph_img = draw_glyph2(selffont, text, contour, scale=1, width=width, height=height)

    # 保存图像（可选）
    if save_path is not None:
        glyph_img_uint8 = (1.0 - glyph_img) * 255  # 转换为白底黑字
        glyph_img_uint8 = glyph_img_uint8.astype(np.uint8)[:, :, 0]
        glyph_pil = Image.fromarray(glyph_img_uint8, mode="L")
        glyph_pil.save(save_path)
        print(f"[✓] Glyph image saved to: {save_path}")
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # 从单通道转为三通道
        mask_save_path = save_path.replace(".png", "_mask_rgb.png")
        cv2.imwrite(mask_save_path, mask_rgb)

    return glyph_img


def main():
    parser = argparse.ArgumentParser(description='从掩码图像生成文字glyph图像')
    parser.add_argument('--mask_path', type=str, required=True,
                       help='输入掩码图像路径')
    parser.add_argument('--text', type=str, required=True,
                       help='要绘制的文字内容')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出glyph图像路径')
    parser.add_argument('--font_path', type=str, 
                       default='/root/paddlejob/workspace/env_run/zhuyinghao/FluxText/font/Arial_Unicode.ttf',
                       help='字体文件路径')
    parser.add_argument('--font_size', type=int, default=60,
                       help='字体大小')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.mask_path):
        print(f"错误：掩码图像文件不存在: {args.mask_path}")
        sys.exit(1)
    
    if not os.path.exists(args.font_path):
        print(f"错误：字体文件不存在: {args.font_path}")
        sys.exit(1)
    
    # 读取掩码图像
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"错误：无法读取掩码图像: {args.mask_path}")
        sys.exit(1)
    
    # 从mask图像获取宽度和高度
    height, width = mask.shape
    print(f"[INFO] 从掩码图像获取尺寸: width={width}, height={height}")
    
    try:
        # 生成glyph图像
        glyph = render_glyph_image(
            mask=mask,
            text=args.text,
            width=width,
            height=height,
            font_path=args.font_path,
            font_size=args.font_size,
            save_path=args.output_path
        )
        print(f"[✓] 成功生成glyph图像: {args.output_path}")
        
    except Exception as e:
        print(f"错误：生成glyph图像失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()