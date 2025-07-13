import os
from PIL import Image

def split_image(input_path, output_dir, tile_size=128):
    """
    将输入图像分割成指定大小的块
    :param input_path: 输入图像路径或包含图像的文件夹
    :param output_dir: 输出目录
    :param tile_size: 分割块大小，默认为128x128
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理单个文件或文件夹
    if os.path.isfile(input_path):
        process_single_image(input_path, output_dir, tile_size)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                process_single_image(file_path, output_dir, tile_size)

def process_single_image(image_path, output_dir, tile_size):
    """处理单个图像文件"""
    try:
        img = Image.open(image_path)
        width, height = img.size
        basename = os.path.splitext(os.path.basename(image_path))[0]

        # 计算行列数
        cols = width // tile_size
        rows = height // tile_size

        # 分割图像
        for i in range(rows):
            for j in range(cols):
                left = j * tile_size
                upper = i * tile_size
                right = left + tile_size
                lower = upper + tile_size
                
                # 裁剪图像块
                tile = img.crop((left, upper, right, lower))
                
                # 保存图像块
                output_path = os.path.join(output_dir, f"{basename}_{i}_{j}.png")
                tile.save(output_path)
                
        print(f"成功处理: {image_path} -> {cols*rows}个块")
    except Exception as e:
        print(f"处理失败 {image_path}: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='图像分割工具')
    parser.add_argument('input_path', help='输入图像路径或包含图像的文件夹')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('--tile_size', type=int, default=128, help='分割块大小，默认为128x128')
    
    args = parser.parse_args()
    split_image(args.input_path, args.output_dir, args.tile_size)