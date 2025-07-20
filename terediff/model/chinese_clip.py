import torch
import torch.nn as nn
from typing import List
try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name, available_models
    from cn_clip.clip import tokenize
    CHINESE_CLIP_AVAILABLE = True
except ImportError:
    print("Chinese-CLIP not available, falling back to original CLIP")
    CHINESE_CLIP_AVAILABLE = False
    from .open_clip import CLIP, tokenize

class ChineseCLIPEmbedder(nn.Module):
    """支持中文的CLIP文本编码器"""
    
    def __init__(self, model_name="ViT-B-16", pretrained="zh"):
        super().__init__()
        if CHINESE_CLIP_AVAILABLE:
            # self.model, self.preprocess = clip.load(model_name, device="cpu", download_root="./weights/")
            self.model, self.preprocess = load_from_name(model_name, device='cpu')
            self.use_chinese_clip = True
            print("Chinese-CLIP loaded successfully")
        else:
            # 回退到原始CLIP
            print("notice: Falling back to original CLIP")
            from .clip import FrozenOpenCLIPEmbedder
            self.model = FrozenOpenCLIPEmbedder()
            self.use_chinese_clip = False
            self.projection = None
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.use_chinese_clip:
            # 使用Chinese-CLIP，但返回序列特征而不是池化特征
            device = next(self.model.parameters()).device
            text_tokens = tokenize(texts).to(device)
            
            with torch.no_grad():
                # 获取transformer的中间特征，类似FrozenOpenCLIPEmbedder的做法
                x = self.model.token_embedding(text_tokens).type(self.model.dtype)
                x = x + self.model.positional_embedding.type(self.model.dtype)
                x = x.permute(1, 0, 2)  # NLD -> LND
                x = self.model.transformer(x)
                x = x.permute(1, 0, 2)  # LND -> NLD
                x = self.model.ln_final(x).type(torch.float32)  # [batch_size, n_ctx, transformer.width]
                
                # 应用投影层到每个token
                if self.projection is not None:
                    # x shape: [batch_size, seq_len, 512] -> [batch_size, seq_len, 1024]
                    x = self.projection(x)
                
            return x
        else:
            # 使用原始CLIP
            return self.model.encode(texts)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


def main():
    """调试中文CLIP编码器"""
    print("=== 调试中文CLIP编码器 ===")
    
    # 测试文本（中英文混合）
    test_texts = ["你好世界", "Hello world", "这是测试文本", "Text encoding test"]
    
    try:
        # 创建模型实例
        model = ChineseCLIPEmbedder(model_name="ViT-B-16", pretrained="zh")
        print(f"模型创建成功，使用Chinese-CLIP: {model.use_chinese_clip}")
        
        # 编码文本
        encoded = model.encode(test_texts)
        print(f"输入文本: {test_texts}")
        print(f"编码输出形状: {encoded.shape}")
        print(f"编码输出数据类型: {encoded.dtype}")
        print(f"编码输出设备: {encoded.device}")
        
        # 检查每个token的维度
        print(f"每个token的特征维度: {encoded.shape[-1]}")
        print(f"序列长度: {encoded.shape[1]}")
        print(f"批次大小: {encoded.shape[0]}")
        
        # 检查模型详细信息
        if model.use_chinese_clip:
            print(f"Chinese-CLIP模型类型: {type(model.model)}")
            print(f"模型dtype: {model.model.dtype}")
            print(f"transformer宽度: {model.model.transformer.width}")
            print(f"token_embedding维度: {model.model.token_embedding.weight.shape}")
            print(f"positional_embedding维度: {model.model.positional_embedding.shape}")
        
        return encoded
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_encoders():
    """比较两个编码器的输出"""
    print("\n=== 比较编码器输出 ===")
    
    # 测试文本
    test_texts = ["Hello world", "Test text"]
    
    try:
        # 测试原始CLIP（使用配置文件参数）
        print("测试原始CLIP...")
        from .clip import FrozenOpenCLIPEmbedder
        
        embed_dim = 1024  # 嵌入维度：最终输出的特征维度
        
        vision_cfg = {
            'image_size': 224,      # 图像尺寸：输入图像的分辨率
            'layers': 32,           # 视觉层数：Vision Transformer的层数
            'width': 1280,          # 视觉宽度：视觉特征的隐藏层维度
            'head_width': 80,       # 注意力头宽度：每个注意力头的维度
            'patch_size': 14        # 图像块大小：将图像分割成的小块尺寸
        }
        
        text_cfg = {
            'context_length': 77,   # 上下文长度：文本序列的最大长度
            'vocab_size': 49408,    # 词汇表大小：词表的总词数
            'width': 1024,          # 文本宽度：文本特征的隐藏层维度
            'heads': 16,            # 注意力头数：多头注意力的头数
            'layers': 24            # 文本层数：文本Transformer的层数
        }
        
        layer = "penultimate"       # 输出层：选择哪一层的特征输出（倒数第二层）
        
        original_model = FrozenOpenCLIPEmbedder(embed_dim, vision_cfg, text_cfg, layer)
        original_output = original_model.encode(test_texts)
        print(f"原始CLIP输出形状: {original_output.shape}")
        
        # 测试中文CLIP
        print("测试中文CLIP...")
        chinese_model = ChineseCLIPEmbedder()
        chinese_output = chinese_model.encode(test_texts)
        print(f"中文CLIP输出形状: {chinese_output.shape}")
        
        # 比较维度
        print(f"\n维度比较:")
        print(f"原始CLIP: {original_output.shape}")
        print(f"中文CLIP: {chinese_output.shape}")
        
        if original_output.shape == chinese_output.shape:
            print("✓ 输出维度一致")
        else:
            print("✗ 输出维度不一致")
            print(f"差异: {original_output.shape} vs {chinese_output.shape}")
            
            # 分析差异
            if len(original_output.shape) == len(chinese_output.shape):
                for i, (orig_dim, chinese_dim) in enumerate(zip(original_output.shape, chinese_output.shape)):
                    if orig_dim != chinese_dim:
                        print(f"第{i}维差异: {orig_dim} vs {chinese_dim}")
                        
                        # 如果是特征维度差异，提供解决方案
                        if i == 2:  # 特征维度
                            print(f"需要添加投影层: {chinese_dim} -> {orig_dim}")
                            print("建议在ChineseCLIPEmbedder中添加投影层")
        
        return original_output, chinese_output
        
    except Exception as e:
        print(f"比较过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def debug_tokenization():
    """调试tokenization过程"""
    print("\n=== 调试Tokenization过程 ===")
    
    test_texts = ["你好世界", "Hello world", "这是测试文本"]
    
    try:
        if CHINESE_CLIP_AVAILABLE:
            # 测试Chinese-CLIP的tokenization
            print("测试Chinese-CLIP tokenization...")
            tokens = tokenize(test_texts)
            print(f"输入文本: {test_texts}")
            print(f"Token形状: {tokens.shape}")
            print(f"Token数据类型: {tokens.dtype}")
            print(f"Token设备: {tokens.device}")
            
            # 显示每个文本的token数量
            for i, text in enumerate(test_texts):
                print(f"文本 '{text}' 的token数量: {tokens[i].shape[0]}")
                
        else:
            print("Chinese-CLIP不可用，无法测试tokenization")
            
    except Exception as e:
        print(f"Tokenization调试出错: {e}")
        import traceback
        traceback.print_exc()


def debug_model_structure():
    """调试模型结构"""
    print("\n=== 调试模型结构 ===")
    
    try:
        model = ChineseCLIPEmbedder()
        
        if model.use_chinese_clip:
            print("Chinese-CLIP模型结构:")
            print(f"模型类型: {type(model.model)}")
            print(f"模型设备: {next(model.model.parameters()).device}")
            print(f"模型dtype: {model.model.dtype}")
            
            # 检查transformer结构
            print(f"Transformer层数: {len(model.model.transformer.resblocks)}")
            print(f"Transformer宽度: {model.model.transformer.width}")
            
            # 检查embedding层
            print(f"Token embedding形状: {model.model.token_embedding.weight.shape}")
            print(f"Positional embedding形状: {model.model.positional_embedding.shape}")
            
            # 检查最终层
            print(f"最终层归一化: {type(model.model.ln_final)}")
            
        else:
            print("使用原始CLIP模型")
            
    except Exception as e:
        print(f"模型结构调试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行所有调试函数
    print("开始调试Chinese-CLIP编码器...")
    
    # 1. 调试tokenization
    debug_tokenization()
    
    # 2. 调试模型结构
    debug_model_structure()
    
    # 3. 单独测试中文CLIP
    chinese_output = main()
    
    # 4. 比较两个编码器
    original_output, chinese_output = compare_encoders()
    
    print("\n=== 调试完成 ===") 