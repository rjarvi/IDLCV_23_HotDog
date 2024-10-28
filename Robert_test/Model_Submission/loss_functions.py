import torch
import torchvision.transforms as TF
import torch.nn.functional as F

from torchmetrics import Dice, JaccardIndex, Accuracy, Recall, Specificity

#  Dice overlap, Intersection overUnion, Accuracy, Sensitivity, and Specifici
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

# def dice_loss(y_true, y_pred):
#     smooth = 1e-6  # To avoid division by zero
#     y_true_f = y_true.view(-1)
#     y_pred_f = y_pred.view(-1)
#     intersection = (y_true_f * y_pred_f).sum()
#     dice = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
#     return 1 - dice


# Define a wrapper function for the Dice loss
# def dice_loss(y_true, y_pred):
#     dice_metric = Dice().to(device)
#     y_pred = y_pred.round().int()
#     y_true = y_true.round().int()
#     dice_value = dice_metric(y_pred, y_true)
#     return 1 - dice_value  # Convert the Dice score to Dice loss

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

def focal_loss(y_real, y_pred,gamma = 2):
    y_pred_sig = torch.sigmoid(y_pred)
    term = (1-y_pred_sig)*gamma * y_real * torch.log(y_pred_sig) + (1-y_real) * torch.log(1-y_pred_sig)
    return (-term.sum()) 




# def dice_coefficient(y_pred, y_true, epsilon=1e-07):
#     y_pred_copy = prediction.clone()

#     y_pred_copy[prediction_copy < 0] = 0
#     y_pred_copy[prediction_copy > 0] = 1

#     intersection = abs(torch.sum(y_pred_copy * y_true))
#     union = abs(torch.sum(y_pred_copy) + torch.sum(y_true))
#     dice = (2. * intersection + epsilon) / (union + epsilon)
#     return dice

# Intersection overUnion

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


def evaluate_model_with_metric(model, device, test_loader, metric_fn):
    model.eval()  # Set model to evaluation mode
    total_metric = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient computation
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(device)  # Move data to the same device as the model (e.g., GPU)
            Y_batch = Y_batch.to(device)

            # Forward pass: Get model predictions
            Y_pred = model(X_batch)

            # Calculate the metric for this batch using the provided metric function
            metric_value = metric_fn(Y_batch, Y_pred)
            total_metric += metric_value.item() * X_batch.size(0)  # Sum the metric over the batch

            total_samples += X_batch.size(0)

    # Compute average metric value over the entire test set
    avg_metric = total_metric / total_samples
    #print(f'Final Model Performance - Average Metric: {avg_metric:.4f}')
    
    return avg_metric



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
        for batch in dataloader:
            # Assuming each batch has images and masks
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass to get predictions
            outputs = model(images)
            
            # Convert outputs to probabilities if necessary
            y_pred = torch.sigmoid(outputs)  # Assuming binary segmentation
            
            # Calculate metrics for this batch
            metrics = calculate_segmentation_metrics(masks, y_pred, device)
            
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


#loss function for abalation study
