import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: class weighting factor
            gamma: focusing parameter to reduce loss for easy examples
            reduction: specifies reduction method"""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: predicted output [bs, num_classes]
        targets: groung truth labels
        """
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # Extract true class prob

        # Compute focal weight (1 - p_t)^gamma
        focal_weight = (1 - target_probs) ** self.gamma

        # Compute Cross-Entropy loss
        ce_loss = -torch.log(target_probs + 1e-8)

        # Apply alpha weighting
        if self.alpha is not None:
            alpha_weight = self.alpha[targets] if isinstance(self.alpha, torch.Tensor) else self.alpha
            ce_loss *= alpha_weight

        # Compute final loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
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