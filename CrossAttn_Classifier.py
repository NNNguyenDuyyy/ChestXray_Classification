import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=512, num_heads=2, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        # Learnable projections to align features from image and text branches.
        self.img_proj = nn.Linear(feature_dim, feature_dim)
        self.text_proj = nn.Linear(feature_dim, feature_dim)
        # MultiheadAttention expects inputs of shape (seq_len, batch, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(feature_dim)
        #self.dropout = nn.Dropout(dropout)
    
    def forward(self, img_feature, text_features):
        """
        img_feature: Tensor of shape (B, 512)   - a single image token feature
        anomaly_features: Tensor of shape (B, 2, 512) - corresponding anomaly tokens (e.g., 'normal' and 'abnormal')
        """
        # Project both modalities to a shared space.
        img_proj = self.img_proj(img_feature)         # shape: (B, 512)
        txt_proj = self.text_proj(text_features)     # shape: (B, 2, 512)
        
        # Reshape for multi-head attention:
        # Treat the image feature as a single query token.
        # nn.MultiheadAttention expects shape (seq_len, batch, embed_dim)
        query = img_proj.unsqueeze(0)       # shape: (1, B, 512)
        key   = txt_proj.transpose(0, 1)      # shape: (2, B, 512)
        value = key                         # shape: (2, B, 512)
        
        attn_output, attn_weights = self.cross_attn(query=query, key=key, value=value)
        # Residual connection + layer norm
        fused = self.norm(query + self.dropout(attn_output)).squeeze(0)  # shape: (B, 512)
        return fused, attn_weights  # attn_weights: (B, 1, 2)

class MultiLabelClassifierWithAnomaly(nn.Module):
    def __init__(self, feature_dim=512, num_heads=8, num_labels=15, dropout=0.1, fusion_type="concat"):
        """
        fusion_type: if "concat", the original image token and fused anomaly information are concatenated.
                     Alternatively, they can be added.
        num_labels: Number of labels for the multi-label classification task.
        """
        super(MultiLabelClassifierWithAnomaly, self).__init__()
        self.fusion = CrossAttentionFusion(feature_dim, num_heads, dropout)
        self.fusion_type = fusion_type
        
        # The classifier input dimension depends on the fusion strategy.
        if fusion_type == "concat":
            self.classifier = nn.Linear(feature_dim * 2, num_labels)
        else:
            self.classifier = nn.Linear(feature_dim, num_labels)
    
    def forward(self, img_features, text_features):
        """
        img_features: Tensor of shape (B, 5, 512)    - e.g., image patch tokens
        text_features: Tensor of shape (B, 5, 2, 512) - anomaly cues for each image token
        """
        B, T, D = img_features.shape  # T=5 tokens
        # Flatten the first two dimensions so that each token is processed independently.
        img_flat = img_features.reshape(B * T, D)         # shape: (B*T, 512)
        text_flat = text_features.reshape(B * T, 2, D)  # shape: (B*T, 2, 512)
        
        # Apply cross-attention fusion on each image token with its corresponding anomaly tokens.
        fused_flat, attn_weights_flat = self.fusion(img_flat, text_flat)  # fused_flat: (B*T, 512)
        # Reshape back to original token dimension.
        fused_tokens = fused_flat.reshape(B, T, D)  # (B, 5, 512)
        
        # Combine the original image features with the fused anomaly features.
        if self.fusion_type == "concat":
            combined = torch.cat([img_features, fused_tokens], dim=-1)  # (B, 5, 1024)
        else:
            combined = img_features + fused_tokens  # (B, 5, 512)
        
        # Pool across tokens (e.g., average over the 5 tokens) to get a global representation.
        pooled = torch.mean(combined, dim=1)  # (B, feature_dim*2) if concat, or (B, feature_dim) if added
        
        # Classification head for multi-label prediction.
        logits = self.classifier(pooled)  # (B, num_labels)
        return logits, attn_weights_flat

# Example usage:
if __name__ == "__main__":
    batch_size = 4
    num_tokens = 5  # 5 image tokens per sample
    feature_dim = 512
    num_labels = 15  # Number of multi-label outputs
    
    # Simulated features:
    # Image features from Encoder B: shape [batch_size, 5, 512]
    img_features = torch.randn(batch_size, num_tokens, feature_dim)
    # Anomaly features from Encoder A (CLIP-based), shape [batch_size, 5, 2, 512]
    text_features = torch.randn(batch_size, num_tokens, 2, feature_dim)
    
    model = MultiLabelClassifierWithAnomaly(feature_dim=feature_dim, num_heads=8, num_labels=num_labels, fusion_type="concat")
    logits, attn_weights = model(img_features, text_features)
    
    print("Logits shape:", logits.shape)  # Expected: (batch_size, num_labels)
    # Note: attn_weights_flat has shape (B*T, 1, 2), one set per token.
    print("Attention weights shape (flattened):", attn_weights.shape)
    
    # For multi-label classification, applying a sigmoid activation is common:
    probs = torch.sigmoid(logits)
    print("Predicted probabilities:", probs)
