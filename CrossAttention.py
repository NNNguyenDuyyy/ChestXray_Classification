import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """
    Cross Attention between anomaly feature and input image feature
    Input: 
        + query: anomaly feature
        + key-value: input image feature
    Output:
        attn_output
    """

    def __init__(self, dim_q, dim_kv, num_heads=8):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(dim_q, dim_q)  # Project query to same dim
        self.key_proj = nn.Linear(dim_kv, dim_q)  # Project key to match query dim
        self.value_proj = nn.Linear(dim_kv, dim_q)  # Project value to match query dim

        self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=num_heads, batch_first=True)

    def forward(self, query, key_value):
        query = self.query_proj(query)
        key = self.key_proj(key_value)
        value = self.value_proj(key_value)

        attn_output, _ = self.attn(query, key, value)
        return attn_output
    
if __name__ == "__main__":
    anomaly_feature = torch.randn(1, 1, 1024)
    input_image_feature = torch.rand(1, 1, 512)

    cross_attn = CrossAttention(dim_q=1024, dim_kv=512)

    output = cross_attn(anomaly_feature, input_image_feature)
    print(f"Output shape: {output.shape}")   # Expected: [1, 1, 1024]