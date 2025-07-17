import torch
import torch.nn as nn
from typing import List
try:
    import cn_clip.clip as clip
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
            self.model, self.preprocess = clip.load(model_name, device="cpu", download_root="./weights/")
            self.use_chinese_clip = True
            print("Chinese-CLIP loaded successfully")
        else:
            # 回退到原始CLIP
            print("notice: Falling back to original CLIP")
            from .clip import FrozenOpenCLIPEmbedder
            self.model = FrozenOpenCLIPEmbedder()
            self.use_chinese_clip = False
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.use_chinese_clip:
            # 使用Chinese-CLIP
            device = next(self.model.parameters()).device
            text_tokens = tokenize(texts).to(device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
            return text_features
        else:
            # 使用原始CLIP
            return self.model.encode(texts)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts) 