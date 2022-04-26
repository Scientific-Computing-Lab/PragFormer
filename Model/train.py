import numpy as np
import time
import torch
from sklearn.metrics import accuracy_score
import os

def train(model, epochs, train_dataloader, device, cross_entropy, optimizer, val_dataloader, save):
    train_losses = []
    valid_losses = []
    # set initial loss to infinite
    best_valid_loss = float('inf')
    result_name = save.split(".pt")[0] + ".txt"
    f = open(result_name, "w")
    f.close()
    # for each epoch
    for epoch in range(epochs):

        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train_epoch(model, train_dataloader, device, cross_entropy, optimizer)

        # evaluate model
        valid_loss, acc_score = evaluate(model, val_dataloader, device, cross_entropy)

        # save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(model.state_dict(), save)

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        print(f'Accuracy: {acc_score:.3f}')
        with open(result_name, "a") as f:
            f.writelines(f'{train_loss:.3f} {valid_loss:.3f} {acc_score:.3f}')
            f.writelines("\n")
        # f = open("results.txt", "a")




def train_epoch(model, train_dataloader, device, cross_entropy, optimizer):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []
    # iterate over batches
    for step, batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

        # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis = 0)

    # returns the loss and predictions
    return avg_loss, total_preds



# function for evaluating the model
def evaluate(model, val_dataloader, device, cross_entropy):
    print("\nEvaluating...")

    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0
    acc_score = 0
    # empty list to save the model predictions
    total_preds = []
    t0 = time.time()
    # iterate over batches
    for step, batch in enumerate(val_dataloader):

        # Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = time.time() - t0

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss = total_loss + loss.item()

            # model predictions are stored on GPU. So, push it to CPU
            preds = preds.detach().cpu().numpy()
            labels_acc = labels.detach().cpu().numpy()

            # print("Accuracy:", accuracy_score(labels_acc, np.argmax(preds, axis = 1)))
            acc_score = acc_score + accuracy_score(labels_acc, np.argmax(preds, axis = 1))
            total_preds.append(preds)
    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader)
    acc_score = acc_score / len(val_dataloader)
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis = 0)

    return avg_loss, acc_score