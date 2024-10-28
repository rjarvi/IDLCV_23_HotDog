from time import time
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def train_with_validation(device, model, opt, loss_fn, epochs, train_loader, val_loader, test_loader, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5, verbose=True)
    model.to(device)
    best_epoch = 0
    best_model = model.state_dict()
    
    for epoch in range(epochs):
        print('* Epoch %d/%d' % (epoch+1, epochs))

        # Training phase
        avg_train_loss = 0
        model.train()  # train mode
        for dictionary in train_loader:
            X_batch = dictionary['image']
            Y_batch = dictionary['vessel_mask']
            mask = dictionary['fov_mask']
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            mask=mask.to(device)
            # Y_batch = (Y_batch > 0).float()
            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)

            Y_pred = Y_pred * mask
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_train_loss += loss.item() / len(train_loader)

        # Validation phase
        avg_val_loss = 0
        model.eval()  # validation mode
        with torch.no_grad():  # disable gradient computation
            for dictionary in train_loader:
                X_batch = dictionary['image'].to(device)
                Y_batch = dictionary['vessel_mask'].to(device)
                mask = dictionary['fov_mask'].to(device)
                Y_val_pred = model(X_batch)
                Y_val_pred = Y_val_pred * mask
                val_loss = loss_fn(Y_batch, Y_val_pred)  # forward-pass
                avg_val_loss += val_loss.item() / len(val_loader)

        print(f' - train loss: {avg_train_loss:.4f} - val loss: {avg_val_loss:.4f}')

        # Adjust learning rate based on validation loss
        lr_scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}, saving model.")
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the model checkpoint
            best_epoch = epoch
            best_model = model
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break
    print("Training complete.")
    print(f"Saving best model: epoch: {best_epoch}, loss: {best_val_loss:.4f}")
    best_model = torch.jit.script(best_model)
    best_model.save("best_model.pt")
    
