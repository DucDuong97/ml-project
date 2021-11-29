# imports a getter for the StrangeSymbol Dataset loader and the test data tensor
import os

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

from dataset import get_strange_symbols_train_loader, get_strange_symbols_train_data, get_strange_symbols_test_data
from make_figures import PATH, FIG_WITDH, FIG_HEIGHT, FIG_HEIGHT_FLAT, setup_matplotlib
from knn import cross_validation, KNN

if __name__ == '__main__':
    setup_matplotlib()

    # executing this prepares a loader, which you can iterate to access the data
    train_x, train_y = get_strange_symbols_train_data()
    dataloader = get_strange_symbols_train_loader(128)

###########################################################

    # hyperparameters
    img_size = 28
    input_size = img_size ** 2
    output_size = 15
    channel_size = 1

    learning_rate = 0.001
    batch_size = 128
    num_epoch = 10
    loss_function = nn.CrossEntropyLoss()

    fc1_size = 512
    fc2_size = 256

    def calculate_output_size(input_size, kernel_size, stride, padding=0):
        return math.floor((input_size + padding*2 - (kernel_size -1) -1)/stride) + 1
    
    conv1_size = 6
    conv1_kernel = 3
    conv1_stride = 1
    maxpool1_size = 2
    conv1_output_size = math.floor(calculate_output_size(img_size, conv1_kernel, conv1_stride)/maxpool1_size)

    conv2_size = 16
    conv2_kernel = 3
    conv2_stride = 2
    maxpool2_size = 1
    conv2_output_size = math.floor(calculate_output_size(conv1_output_size, conv2_kernel, conv2_stride)/maxpool2_size)

    conv3_size = 16
    conv3_kernel = 3
    conv3_stride = 2
    maxpool3_size = 1
    conv3_output_size = math.floor(calculate_output_size(conv2_output_size, conv3_kernel, conv3_stride)/maxpool3_size)

###########################################################

    # Neural Network Models
    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)


    class BatchNorm(nn.Module):
        def __init__(self, C, momentum=0.9):
            super().__init__()
            self.C = C
            self.momentum=momentum
            self.running_mean = torch.zeros(C)
            self.running_var = torch.ones(C)
            self.weight = nn.Parameter(torch.randn(C)/math.sqrt(C))
            self.bias = nn.Parameter(torch.zeros(C))

        def forward(self, X):
            if self.training:
                mean = X.mean([0,2,3])
                var = X.var([0,2,3])
                self.running_mean = self.running_mean*self.momentum + mean*(1-self.momentum)
                self.running_var = self.running_var*self.momentum + var*(1-self.momentum)
            else:
                mean = self.running_mean
                var = self.running_var

            m = mean.view([1, self.C, 1, 1]).expand_as(X)
            v = var.view([1, self.C, 1, 1]).expand_as(X)
            w = self.weight.view([1, self.C, 1, 1]).expand_as(X)
            b = self.bias.view([1, self.C, 1, 1]).expand_as(X)
            out = w * (X - m)/torch.sqrt(v) + b
            return out


    class LNN(nn.Module):
        def __init__(self, input_size, output_size):
            super(LNN, self).__init__()
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
            return LNN(self.input_size, self.output_size)


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
                # nn.Conv2d(conv2_size, conv3_size, kernel_size=conv3_kernel, stride=conv3_stride),
                nn.ReLU(),
                LNN(conv2_size * conv2_output_size * conv2_output_size, output_size)
            )

        def forward(self, x):
            return self.cnn(x)

        def clone(self):
            return CNN(self.channel_size, self.img_size, self.output_size)


    class CNN_BatchNorm(nn.Module):
        def __init__(self, channel_size, img_size, output_size):
            super(CNN_BatchNorm, self).__init__()
            self.channel_size = channel_size
            self.img_size = img_size
            self.output_size = output_size

            self.cnn = nn.Sequential(
                Lambda(lambda x: x.view(-1, channel_size, img_size, img_size).float()),
                BatchNorm(channel_size),
                nn.Conv2d(channel_size, conv1_size, kernel_size=conv1_kernel, stride=conv1_stride),
                nn.ReLU(),
                nn.MaxPool2d(maxpool1_size, maxpool1_size),
                BatchNorm(conv1_size),
                nn.Conv2d(conv1_size, conv2_size, kernel_size=conv2_kernel, stride=conv2_stride),
                nn.ReLU(),
                LNN(conv2_size * conv2_output_size * conv2_output_size, output_size)
            )

        def forward(self, x):
            return self.cnn(x)

        def clone(self):
            return CNN(self.channel_size, self.img_size, self.output_size)


    class CNN_BatchNorm_Builtin(nn.Module):
        def __init__(self, channel_size, img_size, output_size):
            super(CNN_BatchNorm_Builtin, self).__init__()
            self.channel_size = channel_size
            self.img_size = img_size
            self.output_size = output_size

            self.cnn = nn.Sequential(
                Lambda(lambda x: x.view(-1, channel_size, img_size, img_size).float()),
                nn.BatchNorm2d(channel_size),
                nn.Conv2d(channel_size, conv1_size, kernel_size=conv1_kernel, stride=conv1_stride),
                nn.ReLU(),
                nn.MaxPool2d(maxpool1_size, maxpool1_size),
                nn.BatchNorm2d(conv1_size),
                nn.Conv2d(conv1_size, conv2_size, kernel_size=conv2_kernel, stride=conv2_stride),
                nn.ReLU(),
                LNN(conv2_size * conv2_output_size * conv2_output_size, output_size)
            )

        def forward(self, x):
            return self.cnn(x)

        def clone(self):
            return CNN(self.channel_size, self.img_size, self.output_size)


###########################################################

    # Cross Validation

    def dnn_cross_validation(model, X=train_x, Y=train_y, plot_cfm=False, lr=learning_rate, num_epochs=num_epoch, loss_func=loss_function, batch_size=batch_size, m=4):

        report_printed = False
        epoch_losses = []
        epoch_accuracies = []
        fold_accuracies = []

        for fold, (train_idx,val_idx) in enumerate(KFold(n_splits=m,shuffle=True).split(X, Y)):
            print()
            epoch_losses = []
            epoch_accuracies = []

            model = model.clone()
            print('Fold {}'.format(fold + 1))
            dataset = TensorDataset(X, Y)

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #model.to(device)
            optimizer = optim.Adam(model.parameters(), lr)

            # get the result of last epoch to produce analytics
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
                epoch_accuracies.append(acc.item())
                epoch_losses.append(valid_loss)
                
                print(f"Epoch {epoch} - Loss: {round(valid_loss, 2)}, Acc: {round(acc.item(), 2)}")
            
            fold_accuracies.append(epoch_accuracies[-1])

            print("------------------------------")
            # we will plot the result only on the last fold
            if not report_printed:
                report_printed = not report_printed
                confs = torch.cat(confs).numpy()
                imgs = torch.cat(imgs).numpy()
                preds = torch.cat(preds).numpy()
                actuals = torch.cat(actuals).numpy()

                if plot_cfm:
                    plot_confusion_matrix(preds, actuals)
                    plot_confident_imgs(confs, imgs, preds, actuals)
            
            # Run only 1 fold
            break
        return epoch_losses, epoch_accuracies, sum(fold_accuracies)/m


    def loss_batch(model, loss_func, xb, yb, opt=None):
        scores = model(xb)
        m = nn.LogSoftmax(dim=1)
        softmax_scores = m(scores)
        confs,preds = torch.max(softmax_scores, dim=1)
        loss = loss_func(softmax_scores, yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), confs, xb, preds, yb, len(xb)


    def run_with_knn(model, dataloader=dataloader, lr=learning_rate, num_epochs=num_epoch, loss_func=loss_function, m=4):
        model = model.clone()

        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #model.to(device)
        optimizer = optim.Adam(model.parameters(), lr)

        for epoch in range(num_epochs):
            model.train()
            for xb, yb in dataloader:
                scores = model(xb)
                loss = loss_func(scores, yb)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        scores_total = []
        labels_total = []
        for xb, yb in dataloader:    
            model.eval()
            with torch.no_grad():
                scores_total.append(model(xb))
            labels_total.append(yb)
        
        scores_total = torch.cat(scores_total).numpy()
        labels_total = torch.cat(labels_total).numpy()
        return cross_validation(KNN(), scores_total, labels_total, m=m)
                

###########################################################

    # plot result

    def plot_losses_and_accs(labels, losses, accuracies, filename):

        fig, axs = plt.subplots(2, figsize=(FIG_WITDH, FIG_HEIGHT))
        num = len(labels)
        epochs_num = len(losses[0])
        
        for i in range(num):
            axs[0].plot(range(epochs_num), losses[i], label=labels[i])
            axs[1].plot(range(epochs_num), accuracies[i], label=labels[i])
        
        plt.setp(axs[0], ylabel='loss')
        axs[0].set_title('Loss pre Epoch')
        plt.setp(axs[1], ylabel='accuracy')
        axs[1].set_title('Accuracy pre Epoch')
        plt.setp(axs[1], xlabel='epoch')
        axs[0].legend()
        axs[1].legend()

        plt.savefig(os.path.join(PATH, filename))
        plt.close(fig)


    def plot_confusion_matrix(labels, preds):
        plt.rcParams.update({'axes.titlesize': 6})
        plt.figure(figsize=(FIG_WITDH,FIG_HEIGHT))
        cm = confusion_matrix(labels, preds)
        df_cm = DataFrame(cm)
        sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='d')
        plt.savefig(os.path.join(PATH, f'2d_confusion_matrix.pdf'))
        # plt.close(fig)


    def plot_confident_imgs(confs, imgs, preds, actuals):
        plt.rcParams.update({'axes.titlesize': 6})
        classes = np.unique(actuals)
        for clazz in classes:
            zips = zip(confs, imgs, preds,actuals)
            class_member =  list(filter(lambda x: x[2] == clazz,zips))
            class_correct = list(filter(lambda x: x[2] == x[3],class_member))
            correct_most_10_confident = sorted(class_correct, key=lambda x: x[0], reverse=True)[:10]

            class_incorrect = list(filter(lambda x: x[2] != x[3],class_member))
            incorrect_least_10_confident = sorted(class_incorrect, key=lambda x: x[0], reverse=False)[:10]

            fig, axs = plt.subplots(2,5, figsize=(FIG_WITDH, FIG_HEIGHT))
            fig.suptitle(f"Correct, most confident images, Class {clazz}")
            for (conf,img,_,_), ax in zip(correct_most_10_confident, axs.flatten()):
                ax.imshow(img.reshape((28,28)))
                ax.set_title(round(conf,2))
            plt.savefig(os.path.join(PATH, f'2d_confident_imgs_class_{clazz}.pdf'))
            plt.close(fig)

            fig, axs = plt.subplots(2,5, figsize=(FIG_WITDH, FIG_HEIGHT))
            fig.suptitle(f"Incorrect, most unconfident images, Class {clazz}")

            for (conf,img,_,_), bx in zip(incorrect_least_10_confident, axs.flatten()):

                bx.imshow(img.reshape((28,28)))
                bx.set_title(round(conf,2))
            plt.savefig(os.path.join(PATH, f'2d_unconfident_imgs_class_{clazz}.pdf'))
            plt.close(fig)

###########################################################

    # TODO: c
    # labels = ['CrossEntropyLoss', 'NLLLoss']
    # losses, accies,_ = zip(*[dnn_cross_validation(CNN(channel_size, img_size, output_size), train_x, train_y, loss_func=nn.CrossEntropyLoss()),
    #             dnn_cross_validation(CNN(channel_size, img_size, output_size), train_x, train_y, loss_func=nn.NLLLoss())])
    # plot_losses_and_accs(labels,losses, accies, '2c_dnn_loss_epoch.pdf')


    # TODO: d
    # dnn_cross_validation(CNN(channel_size, img_size, output_size), plot_cfm=True)


    # TODO: e

    # x_axis = ['normal CNN','CNN with KNN']
    # y_axis = [dnn_cross_validation(CNN(channel_size, img_size, output_size))[2],
    #             run_with_knn(CNN(channel_size, img_size, fc2_size))]

    # fig = plt.figure(figsize=(FIG_WITDH, FIG_HEIGHT))
    # plt.plot(x_axis, y_axis)
    # plt.ylabel('Accuracy')
    # plt.title('CNN append with KNN')
    # plt.savefig(os.path.join(PATH, f'2e_dnn.pdf'))
    # plt.close(fig)


    # TODO: f 
    labels = ['normal CNN','CNN BatchNorm', 'Built-in Batch Norm']
    losses, accies,_ = zip(*[dnn_cross_validation(CNN(channel_size, img_size, output_size)),
                        dnn_cross_validation(CNN_BatchNorm(channel_size, img_size, output_size)),
                        dnn_cross_validation(CNN_BatchNorm_Builtin(channel_size, img_size, output_size))])
    plot_losses_and_accs(labels,losses, accies, '2f_dnn.pdf')


    # TODO: Competition
    test_data = get_strange_symbols_test_data()[0]
    # print(test_data)
    # test_data = torch.tensor(test_data)

    model = CNN(channel_size, img_size, output_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate)

    for epoch in range(num_epoch):
        model.train()
        for xb, yb in dataloader:
            scores = model(xb)
            loss = loss_function(scores, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    np.savetxt('test_predictions.csv', model(test_data).detach().numpy(), delimiter=',')


    # The code above is just given as a hint, you may change or adapt it.
    # Nevertheless, you are recommended to use the above loader with some batch size of choice.

    # Finally you have to submit, beneath the report, a csv file of predictions for the test data.
    # Extract the testdata using the provided method:
    # Use the network to get predictions (should be of shape 1500 x 15) and export it to csv using e.g. np.savetxt().

