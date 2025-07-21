import torch
import torch.nn as nn
from typing import List
import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

try:
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name, available_models
    from cn_clip.clip import tokenize
    CHINESE_CLIP_AVAILABLE = True
except ImportError:
    print("Chinese-CLIP not available, falling back to original CLIP")
    CHINESE_CLIP_AVAILABLE = False

class ChineseCLIPEmbedder(nn.Module):
    """支持中文的CLIP文本编码器，获取倒数第二层特征"""
    
    def __init__(self, model_name="ViT-B-16", pretrained="zh", layer="penultimate"):
        super().__init__()
        self.layer = layer
        
        if CHINESE_CLIP_AVAILABLE:
            self.model, self.preprocess = load_from_name(model_name, device='cpu')
            self.use_chinese_clip = True
            print("Chinese-CLIP loaded successfully")
            
            # 获取BERT的层数信息
            self.num_layers = self.model.bert.config.num_hidden_layers
            print(f"BERT层数: {self.num_layers}")
            
            if self.layer == "last":
                self.layer_idx = self.num_layers - 1  # 最后一层
            elif self.layer == "penultimate": 
                self.layer_idx = self.num_layers - 2  # 倒数第二层
            else:
                raise NotImplementedError(f"Layer {layer} not implemented")
                
            print(f"将使用第 {self.layer_idx} 层的特征 (layer='{layer}')")
            
        else:
            # 回退到原始CLIP
            print("notice: Falling back to original CLIP")
            from terediff.model.clip import FrozenOpenCLIPEmbedder
            
            # 使用配置文件中的参数创建原始CLIP
            embed_dim = 1024
            vision_cfg = {
                'image_size': 224,
                'layers': 32,
                'width': 1280,
                'head_width': 80,
                'patch_size': 14
            }
            text_cfg = {
                'context_length': 77,
                'vocab_size': 49408,
                'width': 1024,
                'heads': 16,
                'layers': 24
            }
            
            self.model = FrozenOpenCLIPEmbedder(embed_dim, vision_cfg, text_cfg, layer)
            self.use_chinese_clip = False
    
    def encode_with_bert_transformer(self, text_tokens):
        """类似原始CLIP的encode_with_transformer，但使用BERT"""
        # 获取BERT的输入embedding
        bert_model = self.model.bert
        
        # BERT的forward过程，但只到指定层
        embedding_output = bert_model.embeddings(text_tokens) # [3, 52] ->[3, 52, 768]
        
        # 通过transformer层，但只到倒数第二层
        encoder_outputs = embedding_output
        
        for i, layer_module in enumerate(bert_model.encoder.layer):
            if i >= self.layer_idx + 1:  # 只到指定层
                break
            encoder_outputs = layer_module(encoder_outputs)
        
        # 应用layer norm (类似原始CLIP的ln_final)
        if hasattr(bert_model, 'pooler') and hasattr(bert_model.pooler, 'dense'):
            # 使用BERT的最终layer norm
            if i == self.layer_idx:  # 如果是目标层
                # 不需要pooling，直接返回序列特征
                pass
        
        return encoder_outputs
    
    def encode(self, texts: List[str]) -> torch.Tensor:
        if self.use_chinese_clip:
            # 使用Chinese-CLIP，但获取中间层特征而不是最终池化特征
            device = next(self.model.parameters()).device
            text_tokens = tokenize(texts).to(device) # 3, 52
            
            with torch.no_grad():
                # 获取BERT transformer的中间层特征，类似FrozenOpenCLIPEmbedder的做法
                x = self.encode_with_bert_transformer(text_tokens)
                # x shape: [batch_size, seq_len, hidden_size]
                
            return x
        else:
            # 使用原始CLIP
            return self.model.encode(texts)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


def main():
    """调试中文CLIP编码器"""
    print("=== 调试中文CLIP编码器（获取中间层特征）===")
    
    # 测试文本（中英文混合）
    test_texts = ["你好世界", "Hello world", "这是测试文本"]
    
    try:
        # 创建模型实例
        model = ChineseCLIPEmbedder(model_name="ViT-B-16", layer="penultimate")
        print(f"模型创建成功，使用Chinese-CLIP: {model.use_chinese_clip}")
        
        # 编码文本
        encoded = model.encode(test_texts)
        print(f"输入文本: {test_texts}")
        print(f"编码输出形状: {encoded.shape}")
        print(f"编码输出数据类型: {encoded.dtype}")
        print(f"编码输出设备: {encoded.device}")
        
        # 检查维度
        print(f"\n维度分析:")
        print(f"批次大小: {encoded.shape[0]}")
        print(f"序列长度: {encoded.shape[1]}")
        print(f"特征维度: {encoded.shape[2]}")
        
        # 检查模型信息
        if model.use_chinese_clip:
            print(f"\nChinese-CLIP BERT信息:")
            print(f"BERT总层数: {model.num_layers}")
            print(f"使用层索引: {model.layer_idx} (layer='{model.layer}')")
            print(f"BERT隐藏层维度: {model.model.bert.config.hidden_size}")
            print(f"BERT最大序列长度: {model.model.bert.config.max_position_embeddings}")
            print(f"BERT词汇表大小: {model.model.bert.config.vocab_size}")
        
        return encoded
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_bert_structure():
    """调试BERT结构"""
    print("\n=== 调试BERT结构 ===")
    
    try:
        model = ChineseCLIPEmbedder()
        
        if model.use_chinese_clip:
            bert = model.model.bert
            print(f"BERT配置:")
            print(f"  隐藏层维度: {bert.config.hidden_size}")
            print(f"  层数: {bert.config.num_hidden_layers}")
            print(f"  注意力头数: {bert.config.num_attention_heads}")
            print(f"  中间层维度: {bert.config.intermediate_size}")
            print(f"  最大位置编码: {bert.config.max_position_embeddings}")
            print(f"  词汇表大小: {bert.config.vocab_size}")
            
            # 检查层结构
            print(f"\nBERT层结构:")
            print(f"  Embeddings: {type(bert.embeddings)}")
            print(f"  Encoder层数: {len(bert.encoder.layer)}")
            if hasattr(bert, 'pooler'):
                print(f"  Pooler: {type(bert.pooler)}")
                
    except Exception as e:
        print(f"BERT结构调试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("开始调试Chinese-CLIP编码器...")
    
    # 1. 调试BERT结构
    debug_bert_structure()
    
    # 2. 主要测试
    encoded_output = main()
    
    print("\n=== 调试完成 ===")
    
    if encoded_output is not None:
        print(f"\n✅ 成功！Chinese-CLIP编码器输出形状: {encoded_output.shape}")
        print("现在获取的是BERT倒数第二层的序列特征，而不是池化特征。")
    else:
        print("\n❌ 调试失败，需要进一步检查。") 