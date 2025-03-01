import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=512, num_heads=2, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        self.img_proj = nn.Linear(feature_dim, feature_dim)
        self.text_proj = nn.Linear(feature_dim, feature_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, img_feature, text_features):
        """
        img_feature: Tensor of shape (5*B, 512)   - 5 tokens per image
        text_features: Tensor of shape (5*B, 2, 512) - corresponding anomaly tokens
        """
        img_proj = self.img_proj(img_feature)         # shape: (5*B, 512)
        txt_proj = self.text_proj(text_features)      # shape: (5*B, 2, 512)
        
        query = img_proj.unsqueeze(0)                # shape: (1, 5*B, 512)
        key   = txt_proj.transpose(0, 1)             # shape: (2, 5*B, 512)
        value = key                                  # shape: (2, 5*B, 512)
        
        attn_output, attn_weights = self.cross_attn(query=query, key=key, value=value)
        fused = self.norm(query + self.dropout(attn_output)).squeeze(0)  # shape: (5*B, 512)
        return fused, attn_weights  # attn_weights: (5*B, 1, 2)

class MultiLabelClassifierWithAnomaly(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_labels=15, dropout=0.1, fusion_type="concat"):
        super(MultiLabelClassifierWithAnomaly, self).__init__()
        self.fusion = CrossAttentionFusion(feature_dim, num_heads, dropout)
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.classifier = nn.Linear(feature_dim * 2, num_labels)
        else:
            self.classifier = nn.Linear(feature_dim, num_labels)
    
    def forward(self, img_features, text_features):
        """
        img_features: Tensor of shape (5, B, 512)    - 5 tokens per image
        text_features: Tensor of shape (5, B, 2, 512) - anomaly cues for each image token
        """
        T, B, D = img_features.shape  # T=5 tokens
        img_flat = img_features.reshape(T * B, D)         # shape: (5*B, 512)
        text_flat = text_features.reshape(T * B, 2, D)    # shape: (5*B, 2, 512)
        
        fused_flat, attn_weights_flat = self.fusion(img_flat, text_flat)  # fused_flat: (5*B, 512)
        fused_tokens = fused_flat.reshape(T, B, D)  # (5, B, 512)
        
        if self.fusion_type == "concat":
            combined = torch.cat([img_features, fused_tokens], dim=-1)  # (5, B, 1024)
        else:
            combined = img_features + fused_tokens  # (5, B, 512)
        
        pooled = torch.mean(combined, dim=0)  # (B, feature_dim*2) if concat, or (B, feature_dim) if added
        
        logits = self.classifier(pooled)  # (B, num_labels)
        return logits, attn_weights_flat

# Example usage:
if __name__ == "__main__":
    batch_size = 4
    num_tokens = 5  # 5 image tokens per sample
    feature_dim = 512
    num_labels = 15  # Number of multi-label outputs
    
    img_features = torch.randn(num_tokens, batch_size, feature_dim)  # (5, B, 512)
    text_features = torch.randn(num_tokens, batch_size, 2, feature_dim)  # (5, B, 2, 512)
    
    model = MultiLabelClassifierWithAnomaly(feature_dim=feature_dim, num_heads=8, num_labels=num_labels, fusion_type="concat")
    logits, attn_weights = model(img_features, text_features)
    
    print("Logits shape:", logits.shape)  # Expected: (B, num_labels)
    print("Attention weights shape (flattened):", attn_weights.shape)  # (5*B, 1, 2)
    
    probs = torch.sigmoid(logits)
    print("Predicted probabilities:", probs)
