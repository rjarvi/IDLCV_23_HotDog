import torch
import torchvision.transforms as TF
import torch.nn.functional as F

#  Dice overlap, Intersection overUnion, Accuracy, Sensitivity, and Specifici

def bce_loss(y_real, y_pred):
    return torch.mean(y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred)))

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6  # To avoid division by zero
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)


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