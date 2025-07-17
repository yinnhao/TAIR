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