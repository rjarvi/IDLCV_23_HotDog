from time import time
import torch
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def train_with_validation(device, model, opt, loss_fn, epochs, train_loader, val_loader, test_loader, patience=5):
    X_test, Y_test = next(iter(test_loader))
    best_val_loss = float('inf')
    patience_counter = 0
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=3, factor=0.5, verbose=True)

    for epoch in range(epochs):
        tic = time()
        print('* Epoch %d/%d' % (epoch+1, epochs))

        # Training phase
        avg_train_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_batch, Y_pred)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            # calculate metrics to show the user
            avg_train_loss += loss.item() / len(train_loader)

        # Validation phase
        avg_val_loss = 0
        model.eval()  # validation mode
        with torch.no_grad():  # disable gradient computation
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)

                Y_val_pred = model(X_val)
                val_loss = loss_fn(Y_val, Y_val_pred)  # forward-pass
                avg_val_loss += val_loss.item() / len(val_loader)

        toc = time()
        print(f' - train loss: {avg_train_loss:.4f} - val loss: {avg_val_loss:.4f}')

        # Adjust learning rate based on validation loss
        lr_scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}, saving model.")
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the model checkpoint
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

        # Show intermediate results
        # Y_hat = torch.sigmoid(model(X_test.to(device))).detach().cpu()
        # clear_output(wait=True)
        # for k in range(6):
        #     plt.subplot(3, 6, k+1)
        #     plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
        #     plt.title('Real')
        #     plt.axis('off')

        #     plt.subplot(3, 6, k+7)
        #     plt.imshow(Y_hat[k, 0], cmap='gray')
        #     plt.title('Output')
        #     plt.axis('off')
        #     plt.subplot(3, 6, k+13)
        #     plt.imshow(Y_test[k, 0].detach().cpu(), cmap='gray')
        #     plt.title('Ground Truth')
        #     plt.axis('off')
        # plt.suptitle('%d / %d - train loss: %f - val loss: %f' % (epoch+1, epochs, avg_train_loss, avg_val_loss))
        # plt.show()
