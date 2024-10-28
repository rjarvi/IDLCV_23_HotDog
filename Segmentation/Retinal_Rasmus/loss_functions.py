import torch
import torchvision.transforms as TF
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex, BinaryRecall, BinarySpecificity, Dice as BinaryDiceCoefficient

#  Dice overlap, Intersection overUnion, Accuracy, Sensitivity, and Specifici

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


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

def intersection_over_union(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    union = y_true_f.sum() + y_pred_f.sum() - intersection
    return intersection / (union + smooth)

# Accuracy

def accuracy(y_true, y_pred):
    y_pred = y_pred.round()  # Convert probabilities to binary
    correct = (y_true == y_pred).float()  # Check if predictions are correct
    return correct.sum() / correct.numel()

# Sensitivity (measures true positives)

def sensitivity(y_true, y_pred):
    y_pred = y_pred.round()  # Convert probabilities to binary
    true_positives = (y_true * y_pred).sum()
    possible_positives = y_true.sum()
    return true_positives / (possible_positives + 1e-6)  # Avoid division by zero

# Specificity(measures true negatives)

def specificity(y_true, y_pred):
    y_pred = y_pred.round()  # Convert probabilities to binary
    true_negatives = ((1 - y_true) * (1 - y_pred)).sum()
    possible_negatives = (1 - y_true).sum()
    return true_negatives / (possible_negatives + 1e-6)  # Avoid division by zero

def calculate_segmentation_metrics(y_true,y_pred, device):
    y_pred = y_pred.round().to(device)
    y_true = y_true.long().to(device)
    #dice_score2 = Dice(y_pred, y_true)

    # BinaryDice calculates image by image the Dice Coefficient and averages over the whole batch or tensor.
    # Same thing with all metrics


    # Instantiate metric objects
    dice_metric = BinaryDiceCoefficient().to(device)
    iou_metric = BinaryJaccardIndex().to(device)
    accuracy_metric = BinaryAccuracy().to(device)
    sensitivity_metric = BinaryRecall().to(device)
    specificity_metric = BinarySpecificity().to(device)

    dice_score  = dice_metric(y_pred,y_true)
    iou_score = iou_metric(y_pred,y_true)
    accuracy_score = accuracy_metric(y_pred,y_true)
    sensitivity_score = sensitivity_metric(y_pred,y_true)
    specificity_score = specificity_metric(y_pred,y_true)

    metrics = {
        'dice': dice_score,
        'iou': iou_score,
        'accuracy': accuracy_score,
        'sensitivity': sensitivity_score,
        'specificity': specificity_score
    }
    return metrics


def evaluate_model_with_metric(model, device, test_loader):
    model.eval()  # Set model to evaluation mode
    total_samples = 0
    metrics = {
            'dice': 0.,
            'iou': 0.,
            'accuracy': 0.,
            'sensitivity': 0.,
            'specificity': 0.
        }
    
    
    
    with torch.no_grad():  # Disable gradient computation
        for dictionary in test_loader:
            X_batch = dictionary['image'].to(device)
            Y_batch = dictionary['vessel_mask'].to(device)
            mask = dictionary['fov_mask'].to(device)
            # Forward pass: Get model predictions
            Y_pred = model(X_batch)
            Y_pred = Y_pred * mask
            Y_pred = torch.sigmoid(Y_pred)
            
            # Calculate the metric for this batch using the provided metric function
            metric_value = calculate_segmentation_metrics(Y_batch, Y_pred, device)
            for key in metrics.keys():
                metrics[key] += metric_value[key]* X_batch.size(0)

            total_samples += X_batch.size(0)

    # Compute average metric value over the entire test set
    for key in metrics.keys():
        metrics[key] += metric_value[key] / total_samples
    #print(f'Final Model Performance - Average Metric: {avg_metric:.4f}')
    
    return metrics



