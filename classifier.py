import torch
import torch.nn as nn

class Classifier(nn.Module):
    """
    Classify label
    Input: 
        + input_dim: dimension of input
        + num_classes: number of classes
    """

    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)
        return self.fc(x)
    
if __name__ == "__main__":
    num_classes = 15
    classifier = Classifier(input_dim=1024, num_classes=num_classes)

    output_attn = torch.rand(1, 1, 1024)

    output_classify = classifier(output_attn)

    print(f"Output shape: {output_classify.shape}")   # [bs, num_classes]

    probs = torch.softmax(output_classify, dim=-1)
    print(f"Class Probabilities: {probs}")