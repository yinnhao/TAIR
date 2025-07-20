from typing import List
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .open_clip import CLIP, tokenize


class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, embed_dim, vision_cfg, text_cfg, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        # model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        model = CLIP(embed_dim, dict(vision_cfg), dict(text_cfg))
        del model.visual
        self.model = model
        
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def forward(self, tokens):
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text: List[str]) -> torch.Tensor:
        # convert a batch of text to tensor
        tokens = tokenize(text)
        # move tensor to model device
        tokens = tokens.to(next(self.model.parameters()).device)
        return self(tokens)


def main():
    """调试原始CLIP编码器"""
    print("=== 调试原始CLIP编码器 ===")
    
    # 测试文本
    test_texts = ["Hello world", "This is a test", "Text encoding"]
    
    try:
        # 使用配置文件中的参数
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
        
        model = FrozenOpenCLIPEmbedder(embed_dim, vision_cfg, text_cfg, layer)
        print(f"模型创建成功")
        print(f"使用配置: embed_dim={embed_dim}, layer={layer}")
        print(f"text_cfg: {text_cfg}")
        
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
        print(f"模型text_cfg.width: {text_cfg['width']}")
        print(f"模型embed_dim: {embed_dim}")
        
        return encoded
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
