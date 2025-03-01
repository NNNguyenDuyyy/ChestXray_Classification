import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualCrossAttentionModel(nn.Module):
    def __init__(self, embed_dim=1024, num_labels=14, num_heads=8):
        super(MutualCrossAttentionModel, self).__init__()
        self.embed_dim = embed_dim
        
        # Layer normalization
        self.anomaly_norm = nn.LayerNorm(embed_dim)
        self.global_norm = nn.LayerNorm(embed_dim)
        
        # Self-attention for anomaly features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False
        )
        
        # Cross-attention: Anomaly to Global
        self.anomaly_to_global_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False
        )
        
        # Cross-attention: Global to Anomaly
        self.global_to_anomaly_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=False
        )
        
        # Final classification layer
        self.classifier = nn.Linear(embed_dim * 2, num_labels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, anomaly_feature, global_image_feature):
        """
        Args:
            anomaly_feature: Tensor of shape [bs, 5, embed_dim]
            global_image_feature: Tensor of shape [bs, embed_dim]
        Returns:
            output: Tensor of shape [bs, num_labels]
        """
        # Ensure global_image_feature has the right shape
        if global_image_feature.dim() == 2:
            global_image_feature = global_image_feature.unsqueeze(1)  # [bs, embed_dim] -> [bs, 1, embed_dim]
        
        # 1. Self-Attention on Anomaly Feature
        anomaly_norm = self.anomaly_norm(anomaly_feature)
        anomaly_self_attended = self._attention_forward(
            self.self_attention,
            anomaly_norm, anomaly_norm, anomaly_norm,
            need_weights=False
        )
        anomaly_feature_self_attended = anomaly_feature + anomaly_self_attended
        
        # 2. Cross Attention: Anomaly to Global
        global_norm = self.global_norm(global_image_feature)
        anomaly_to_global_attended = self._attention_forward(
            self.anomaly_to_global_attention,
            anomaly_feature_self_attended, global_norm, global_norm,
            need_weights=False
        )
        enhanced_anomaly_feature = anomaly_feature_self_attended + anomaly_to_global_attended
        
        # 3. Cross Attention: Global to Anomaly
        global_to_anomaly_attended = self._attention_forward(
            self.global_to_anomaly_attention,
            global_norm, enhanced_anomaly_feature, enhanced_anomaly_feature,
            need_weights=False
        )
        enhanced_global_image_feature = global_image_feature + global_to_anomaly_attended
        
        # 4. Pooling and Feature Combination
        pooled_enhanced_anomaly_feature = enhanced_anomaly_feature.mean(dim=1)  # [bs, embed_dim]
        enhanced_global_image_feature_squeezed = enhanced_global_image_feature.squeeze(1)  # [bs, embed_dim]
        
        concatenated_feature = torch.cat([
            pooled_enhanced_anomaly_feature, 
            enhanced_global_image_feature_squeezed
        ], dim=1)  # [bs, 2*embed_dim]
        
        # Apply dropout for regularization
        concatenated_feature = self.dropout(concatenated_feature)
        
        # 5. Classification
        logits = self.classifier(concatenated_feature)
        output = torch.sigmoid(logits)
        
        return output
    
    def _attention_forward(self, attention_module, query, key, value, need_weights=False):
        """
        Helper function to handle the transposition required by PyTorch's MultiheadAttention
        
        Args:
            attention_module: MultiheadAttention module
            query: Tensor of shape [bs, seq_len_q, embed_dim]
            key: Tensor of shape [bs, seq_len_k, embed_dim]
            value: Tensor of shape [bs, seq_len_v, embed_dim]
            need_weights: Whether to return attention weights
            
        Returns:
            output: Tensor of shape [bs, seq_len_q, embed_dim]
        """
        # Transpose for MultiheadAttention: [bs, seq_len, embed_dim] -> [seq_len, bs, embed_dim]
        query_t = query.transpose(0, 1)
        key_t = key.transpose(0, 1)
        value_t = value.transpose(0, 1)
        
        # Apply attention
        if need_weights:
            output, attn_weights = attention_module(query_t, key_t, value_t)
            # Transpose back: [seq_len, bs, embed_dim] -> [bs, seq_len, embed_dim]
            return output.transpose(0, 1), attn_weights
        else:
            output = attention_module(query_t, key_t, value_t)[0]
            # Transpose back: [seq_len, bs, embed_dim] -> [bs, seq_len, embed_dim]
            return output.transpose(0, 1)


# Demo function to test the model
def demo():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Parameters
    batch_size = 4
    embed_dim = 1024
    num_labels = 14
    
    # Create random input tensors
    anomaly_feature = torch.randn(batch_size, 5, embed_dim)
    global_image_feature = torch.randn(batch_size, embed_dim)
    
    # Initialize model
    model = MutualCrossAttentionModel(embed_dim=embed_dim, num_labels=num_labels)
    
    # Forward pass
    output = model(anomaly_feature, global_image_feature)
    
    print(f"Input shapes:")
    print(f"  - anomaly_feature: {anomaly_feature.shape}")
    print(f"  - global_image_feature: {global_image_feature.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (predictions):")
    print(output)
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model

if __name__ == "__main__":
    model = demo()