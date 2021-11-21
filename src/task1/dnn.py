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

    # hyperparameters
    img_size = 28
    input_size = img_size ** 2
    output_size = 15
    channel_size = 1

    learning_rate = 0.01
    batch_size = 128
    num_epoch = 4
    loss_function = nn.CrossEntropyLoss()

    fc1_size = 256
    fc2_size = 64

    def calculate_output_size(input_size, kernel_size, stride, padding=0):
        return math.floor((input_size + padding*2 - (kernel_size -1) -1)/stride) + 1
    
    conv1_size = 6
    conv1_kernel = 3
    conv1_stride = 2
    maxpool1_size = 2
    conv1_output_size = math.floor(calculate_output_size(img_size, conv1_kernel, conv1_stride)/maxpool1_size)

    conv2_size = 16
    conv2_kernel = 3
    conv2_stride = 2
    maxpool2_size = 1
    conv2_output_size = math.floor(calculate_output_size(conv1_output_size, conv2_kernel, conv2_stride)/maxpool2_size)


    # Neural Network Models
    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)

    class DNN(nn.Module):
        def __init__(self, input_size, output_size):
            super(DNN, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.dnn = nn.Sequential(
                Lambda(lambda x: x.view(x.size(0), -1).float()),
                nn.Linear(input_size, fc1_size),
                nn.ReLU(),
                nn.Linear(fc1_size, fc2_size),
                nn.ReLU(),
                nn.Linear(fc2_size, output_size)
            )

        def forward(self, x):
            return self.dnn(x)

        def clone(self):
            return DNN(self.input_size, self.output_size)

    class CNN(nn.Module):
        def __init__(self, channel_size, img_size, output_size):
            super(CNN, self).__init__()
            self.channel_size = channel_size
            self.img_size = img_size
            self.output_size = output_size

            self.cnn = nn.Sequential(
                Lambda(lambda x: x.view(-1, channel_size, img_size, img_size).float()),
                nn.Conv2d(channel_size, conv1_size, kernel_size=conv1_kernel, stride=conv1_stride),
                nn.ReLU(),
                nn.MaxPool2d(maxpool1_size, maxpool1_size),
                nn.Conv2d(conv1_size, conv2_size, kernel_size=conv2_kernel, stride=conv2_stride),
                nn.ReLU(),
                DNN(conv2_size * conv2_output_size * conv2_output_size, output_size)
            )

        def forward(self, x):
            return self.cnn(x)

        def clone(self):
            return CNN(self.channel_size, self.img_size, self.output_size)


    def cross_validation(model, X, Y, lr, num_epochs, loss_func, batch_size, m=4):

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
                # plot_confident_imgs(confs, imgs, preds, actuals)


    def loss_batch(model, loss_func, xb, yb, opt=None):
        scores = model(xb)
        softmax_scores = torch.nn.functional.softmax(scores)
        preds = torch.argmax(softmax_scores, dim=1)
        confs,_ = torch.max(softmax_scores, dim=1)
        loss = loss_func(scores, yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), confs, xb, preds, yb, len(xb)


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


    # cross_validation(DNN(input_size,output_size), train_x, train_y, learning_rate, num_epoch, loss_function, batch_size)
    cross_validation(CNN(channel_size, img_size, output_size), train_x, train_y, learning_rate, num_epoch, loss_function, batch_size)

    # The code above is just given as a hint, you may change or adapt it.
    # Nevertheless, you are recommended to use the above loader with some batch size of choice.

    # Finally you have to submit, beneath the report, a csv file of predictions for the test data.
    # Extract the testdata using the provided method:
    # TODO
    # Use the network to get predictions (should be of shape 1500 x 15) and export it to csv using e.g. np.savetxt().

    # If you encounter problems during this task, please do not hesitate to ask for help!
    # Please check beforehand if you have installed all necessary packages found in requirements.txt
