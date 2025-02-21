import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class VitModel(nn.Module):
    """
    Input: tensor (bsxCxHxW)
    Output: features
    """

    def __init__(self, base_model):
        super(VitModel, self).__init__()
        self.model = base_model
        self.model.heads = nn.Identity()  # Remove classification head

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    base_model = vit_b_16(pretrained=True)
    vit_model = VitModel(base_model)

    dummy_tensor = torch.randn(1, 3, 224, 224) # bsxCxHxW

    with torch.no_grad():
        features = vit_model(dummy_tensor)

    print("Feature shape: ", features.shape)  # Expected output: [1, 768]