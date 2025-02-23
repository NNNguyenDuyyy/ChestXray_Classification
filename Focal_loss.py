import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: class weighting factor
            gamma: focusing parameter to reduce loss for easy examples
            reduction: specifies reduction method"""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        logits: predicted output [bs, num_classes]
        targets: groung truth labels
        """
        prob = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)

        # Compute focal weight (1 - p_t)^gamma
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else: 
            return focal_loss
    
if __name__ == "__main__":
    # Example logits and ground truth labels
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.2, 3.0, 0.1]], requires_grad=True)  # Shape: [batch_size, num_classes]
    targets = torch.tensor([0, 1])  # Ground truth labels (shape: [batch_size])

    # Define Focal Loss with class balance
    num_classes = 3
    alpha = torch.tensor([0.25, 0.5, 0.25])  # Class weights (can be None)
    focal_loss_fn = FocalLoss(alpha=alpha, gamma=2.0, reduction='mean')

    # Compute loss
    loss = focal_loss_fn(logits, targets)
    print("Focal Loss:", loss.item())