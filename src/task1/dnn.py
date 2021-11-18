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
            self.fc1 = nn.Linear(input_layer, 256)
            self.fc2 = nn.Linear(256, output_layer)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def clone(self):
            return DNN(self.input_layer, self.output_layer)

    # Hyperparameters
    input_size = 784
    output_size = 15
    learning_rate = 0.001
    batch_size = 128
    num_epoch = 4

    def cross_validation(model, X, Y, lr, num_epochs, m=4):

        report_printed = False

        for fold, (train_idx,val_idx) in enumerate(KFold(n_splits=m,shuffle=True).split(X, Y)):

            model = model.clone()
            print('Fold {}'.format(fold + 1))
            dataset = TensorDataset(X,Y)

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            optimizer = optim.Adam(model.parameters(), lr)
            loss_func = nn.CrossEntropyLoss()

            history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

            # get the result of last epoch to produce analytics
            preds = []
            actuals = []
            nums = None
            for epoch in range(num_epochs):
                model.train()
                for xb, yb in train_loader:
                    loss_batch(model, loss_func, xb, yb, optimizer)
                
                model.eval()
                with torch.no_grad():
                    losses, preds, actuals, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in test_loader])
                valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
                acc = (torch.cat(preds) == torch.cat(actuals)).float().mean()
                print(f"Epoch {epoch} - Loss: {valid_loss}, Acc: {acc}")

            # we will plot the result only on the last fold
            if not report_printed and fold == m-1:
                plot_report(torch.cat(preds), torch.cat(actuals))
                report_printed = not report_printed

    
    def plot_report(labels, preds):
        plt.figure(figsize=(12,8))
        cm = confusion_matrix(labels, preds)
        df_cm = DataFrame(cm)
        sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='d')
        plt.show()


    def loss_batch(model, loss_func, xb, yb, opt=None):
        xb = xb.reshape(xb.shape[0], -1).float() #shape 128 x 784
        scores = model(xb)
        preds = torch.argmax(scores, dim=1)
        loss = loss_func(scores, yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), preds, yb, len(xb)


    cross_validation(DNN(input_size,output_size), train_x, train_y, learning_rate, num_epoch)

    # The code above is just given as a hint, you may change or adapt it.
    # Nevertheless, you are recommended to use the above loader with some batch size of choice.

    # Finally you have to submit, beneath the report, a csv file of predictions for the test data.
    # Extract the testdata using the provided method:
    # TODO
    # Use the network to get predictions (should be of shape 1500 x 15) and export it to csv using e.g. np.savetxt().

    # If you encounter problems during this task, please do not hesitate to ask for help!
    # Please check beforehand if you have installed all necessary packages found in requirements.txt
