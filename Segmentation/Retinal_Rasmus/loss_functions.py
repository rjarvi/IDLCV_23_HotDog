import torch
import torchvision.transforms as TF
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryJaccardIndex, BinaryRecall, BinarySpecificity, Dice as BinaryDiceCoefficient
from torchmetrics import Dice, JaccardIndex, Accuracy, Recall, Specificity

#  Dice overlap, Intersection overUnion, Accuracy, Sensitivity, and Specifici
def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))


# def bce_loss(y_real, y_pred, pos_weight=1.1):
#     # Applying the pos_weight to the positive class in the BCE formula
#     loss = y_pred - y_real * y_pred * pos_weight + torch.log(1 + torch.exp(-y_pred))
#     return torch.mean(loss)

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

# def dice_loss(y_real, y_pred, pos_weight=1.0, epsilon=1e-6):
#     y_pred = torch.sigmoid(y_pred)
#     y_real = y_real.float()
    
#     # Flatten spatial dimensions (keep batch dimension)
#     y_real_flat = y_real.view(y_real.size(0), -1)
#     y_pred_flat = y_pred.view(y_pred.size(0), -1)
    
#     # Apply pos_weight to the intersection term
#     intersection = (y_real_flat * y_pred_flat * pos_weight).sum(dim=1)
#     union = (y_real_flat * pos_weight).sum(dim=1) + y_pred_flat.sum(dim=1)
    
#     # Compute per-sample Dice coefficient
#     dice_coeff = (2. * intersection + epsilon) / (union + epsilon)
    
#     # Compute Dice loss
#     dice_loss = 1. - dice_coeff
#     return dice_loss.mean()

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



def calculate_segmentation_metrics(y_true,y_pred, device):
    y_pred = (y_pred > 0.5).long()  # Convert probabilities to binary
    y_true = y_true.long() 
    #dice_score2=Dice(y_pred, y_true)
    dice_func = Dice().to(device)
    dice_score  = dice_func(y_pred,y_true)
    iou_func = JaccardIndex(task="binary").to(device)
    iou_score = iou_func(y_pred,y_true)
    accuracy_func = Accuracy(task="binary").to(device)
    accuracy_score = accuracy_func(y_pred,y_true)
    recall_func = Recall(task="binary").to(device)
    sensitivity_score=recall_func(y_pred,y_true)
    specificity_func = Specificity(task="binary").to(device)
    specificity_score=specificity_func(y_pred,y_true)
    print (dice_score)
    print(iou_score)
    print(accuracy_score)
    print(sensitivity_score)
    print(specificity_score)
    metrics = {
            'Dice': dice_score,
            'IoU': iou_score,
            'Accuracy': accuracy_score,
            'Sensitivity': sensitivity_score,
            'Specificity': specificity_score
        }
    return metrics


def evaluate_model(model, dataloader, device):
    # Put model in evaluation mode
    model.eval()
    
    # Initialize lists to store metrics for each batch
    dice_scores = []
    iou_scores = []
    accuracy_scores = []
    sensitivity_scores = []
    specificity_scores = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for dictionary in dataloader:
            X_batch = dictionary['image'].to(device)
            Y_batch = dictionary['vessel_mask'].to(device)
            mask = dictionary['fov_mask'].to(device)
            # Forward pass to get predictions
            outputs = model(X_batch)
            outputs = outputs*mask
            # Convert outputs to probabilities if necessary
            y_pred = torch.sigmoid(outputs)  # Assuming binary segmentation
            
            # Calculate metrics for this batch
            metrics = calculate_segmentation_metrics(Y_batch, y_pred, device)
            
            # Append each metric
            dice_scores.append(metrics['Dice'].item())
            iou_scores.append(metrics['IoU'].item())
            accuracy_scores.append(metrics['Accuracy'].item())
            sensitivity_scores.append(metrics['Sensitivity'].item())
            specificity_scores.append(metrics['Specificity'].item())
    
    # Compute average for each metric
    avg_metrics = {
        'Dice': sum(dice_scores) / len(dice_scores),
        'IoU': sum(iou_scores) / len(iou_scores),
        'Accuracy': sum(accuracy_scores) / len(accuracy_scores),
        'Sensitivity': sum(sensitivity_scores) / len(sensitivity_scores),
        'Specificity': sum(specificity_scores) / len(specificity_scores)
    }
    
    return avg_metrics


def binary_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):

    probs = torch.sigmoid(inputs)


    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')


    pt = torch.where(targets == 1, probs, 1 - probs)  # p if y=1, 1-p otherwise
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss


    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss 
