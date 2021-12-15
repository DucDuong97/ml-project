# imports a getter for the StrangeSymbol Dataset loader and the test data tensor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
import numpy as np
import math

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt

from dataset import get_strange_symbols_train_loader, get_strange_symbols_train_data

if __name__ == '__main__':

    # executing this prepares a loader, which you can iterate to access the data
    train_x, train_y = get_strange_symbols_train_data()


    # Now it's up to you to define the network and use the data to train it.
    class DNN(nn.Module):
        def __init__(self, input_layer, output_layer):
            super(DNN, self).__init__()
            self.input_layer = input_layer
            self.output_layer = output_layer
            self.fc1 = nn.Linear(input_layer, 64)
            self.fc2 = nn.Linear(64, 256)
            self.fc3 = nn.Linear(256, output_layer)


        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            #x = F.relu(self.fc3(x))
            x = self.fc3(x)
            return x

        def clone(self):
            return DNN(self.input_layer, self.output_layer)


    class CNN(nn.Module):
        def __init__(self, input_layer, output_layer):
            super(CNN, self).__init__()
            self.input_layer = input_layer
            self.output_layer = output_layer
            self.conv1 = nn.Conv2d(input_layer, 32, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, output_layer, kernel_size=3, stride=2, padding=1)


        def forward(self, x):
            x = x.view(-1, 1, 28, 28)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))


            x = F.avg_pool2d(x, 4)
            return x

        def clone(self):
            return CNN(self.input_layer, self.output_layer)


    # Param-, hyperparameters
    input_size = 784
    output_size = 15
    learning_rate = 0.01
    batch_size = 128
    num_epoch = 8

    def cross_validation(model, X, Y, lr, num_epochs, m=2):

        report_printed = False

        for fold, (train_idx,val_idx) in enumerate(KFold(n_splits=m,shuffle=True).split(X, Y)):
            print()

            model = model.clone()
            print('Fold {}'.format(fold + 1))
            dataset = TensorDataset(X, Y)

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr)
            loss_func = nn.CrossEntropyLoss()

            # get the result of last epoch to produce analytics
            epoch_losses = []
            epoch_accuracies = []
            confs = []
            imgs = []
            preds = []
            actuals = []
            nums = None
            for epoch in range(num_epochs):
                model.train()
                for xb, yb in train_loader:
                    loss_batch(model, loss_func, xb, yb, optimizer)
                
                model.eval()
                with torch.no_grad():
                    losses, confs, imgs, preds, actuals, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in test_loader])
                valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                acc = (torch.cat(preds) == torch.cat(actuals)).float().mean()
                epoch_accuracies.append(acc)
                epoch_losses.append(valid_loss)
                print(f"Epoch {epoch} - Loss: {round(valid_loss, 2)}, Acc: {round(acc.item(), 2)}")

            print("------------------------------")
            # we will plot the result only on the last fold
            if not report_printed and fold == m-1:
                report_printed = not report_printed
                confs = torch.cat(confs).numpy()
                imgs = torch.cat(imgs).numpy()
                preds = torch.cat(preds).numpy()
                actuals = torch.cat(actuals).numpy()

                # plot_losses_and_accs(epoch_losses, epoch_accuracies)
                # plot_report(preds, actuals)
                plot_confident_imgs(confs, imgs, preds, actuals)


    def plot_losses_and_accs(losses, accuracies):
        _, axs = plt.subplots(2, figsize=(8, 6), sharex=True)
        axs[0].plot(range(len(losses)), losses)
        plt.setp(axs[0], ylabel='loss')
        axs[0].set_title('Loss pre Epoch')
        axs[1].plot(range(len(accuracies)), accuracies)
        plt.setp(axs[1], ylabel='accuracy')
        plt.setp(axs[1], xlabel='epoch')
        axs[1].set_title('Accuracy pre Epoch')
        plt.show()
    
    def plot_report(labels, preds):
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(labels, preds)
        df_cm = DataFrame(cm)
        sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='d')
        plt.show()

    def plot_confident_imgs(confs, imgs, preds, actuals):
        classes = np.unique(actuals)
        for clazz in classes:
            zips = zip(confs, imgs, preds,actuals)
            class_member =  list(filter(lambda x: x[2] == clazz,zips))
            class_correct = list(filter(lambda x: x[2] == x[3],class_member))
            correct_most_10_confident = sorted(class_correct, key=lambda x: x[0], reverse=True)[:10]

            class_incorrect = list(filter(lambda x: x[2] != x[3],class_member))
            incorrect_least_10_confident = sorted(class_incorrect, key=lambda x: x[0], reverse=False)[:10]

            fig, axs = plt.subplots(2,10, figsize=(12, 6), sharey=True)
            fig.suptitle(f"Class {clazz}")
            for (conf,img,_,_), ax in zip(correct_most_10_confident, axs[0]):
                ax.imshow(img.reshape((28,28)))
                ax.set_title(round(conf,2))
            axs[0,0].set_ylabel("Correct most Conf")

            for (conf,img,_,actual), bx in zip(incorrect_least_10_confident, axs[1]):
                bx.imshow(img.reshape((28,28)))
                bx.set_title(round(conf,2))
                bx.set_xlabel(actual)
            axs[1,0].set_ylabel("Incorrect least Conf")
            plt.show()
                


    def loss_batch(model, loss_func, xb, yb, opt=None):
        xb = xb.reshape(xb.shape[0], -1).float()  # shape 128 x 784
        scores = torch.squeeze(model(xb))
        softmax_scores = torch.nn.functional.softmax(scores)
        preds = torch.argmax(softmax_scores, dim=1)
        confs,_ = torch.max(softmax_scores, dim=1)
        loss = loss_func(scores, yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), confs, xb, preds, yb, len(xb)


    cross_validation(DNN(input_size,output_size), train_x, train_y, learning_rate, num_epoch)
    # cross_validation(CNN(1, output_size), train_x, train_y, learning_rate, num_epoch)

    # The code above is just given as a hint, you may change or adapt it.
    # Nevertheless, you are recommended to use the above loader with some batch size of choice.

    # Finally you have to submit, beneath the report, a csv file of predictions for the test data.
    # Extract the testdata using the provided method:
    # TODO
    # Use the network to get predictions (should be of shape 1500 x 15) and export it to csv using e.g. np.savetxt().

    # If you encounter problems during this task, please do not hesitate to ask for help!
    # Please check beforehand if you have installed all necessary packages found in requirements.txt
