import torch
import torchvision.transforms as TF
import torch.nn.functional as F


def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def focal_loss(y_real, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    """
    # Apply sigmoid to logits to get probabilities
    
    
    # Flatten the tensors
    y_real = y_real.reshape(-1)
    y_pred = y_pred.reshape(-1)

    # Compute the binary cross-entropy (BCE) loss
    loss = F.binary_cross_entropy_with_logits(y_pred, y_real, reduction="none")

    # Compute the focal loss factor (1 - pt)^gamma
    pt = torch.where(y_real == 1, y_pred, 1 - y_pred)  # p_t = y_pred for positive class, 1-y_pred for negative class
    focal_weight = (1 - pt) ** gamma

    # Apply alpha weighting for the minority class
    alpha_weight = torch.where(y_real == 1, alpha, 1 - alpha)

    # Final focal loss
    loss = focal_weight * alpha_weight * loss

    return loss.mean()

def focal_loss(y_real, y_pred,gamma = 2):
    y_pred_sig = torch.sigmoid(y_pred)
    term = (1-y_pred_sig)**gamma * y_real * torch.log(y_pred_sig) + (1-y_real) * torch.log(1-y_pred_sig)
    return (-term.mean())

def dice_loss(y_real, y_pred, epsilon=1e-6):
    y_pred = torch.sigmoid(y_pred)
    y_real = y_real.float()
    
    # Flatten spatial dimensions (keep batch dimension)
    y_real_flat = y_real.view(y_real.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    
    # Compute per-sample Dice coefficient
    intersection = (y_real_flat * y_pred_flat).sum(dim=1)
    union = y_real_flat.sum(dim=1) + y_pred_flat.sum(dim=1)
    dice_coeff = (2. * intersection + epsilon) / (union + epsilon)
    
    # Compute Dice loss
    dice_loss = 1. - dice_coeff
    return dice_loss.mean()