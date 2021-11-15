# imports a getter for the StrangeSymbol Dataset loader and the test data tensor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from pandas import DataFrame
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_strange_symbols_train_loader, get_strange_symbols_test_data

if __name__ == '__main__':
    # executing this prepares a loader, which you can iterate to access the data
    trainloader = get_strange_symbols_train_loader(batch_size=128)


    # TODO
    # Now it's up to you to define the network and use the data to train it.
    class DNN(nn.Module):
        def __init__(self, input_layer, output_layer):
            super(DNN, self).__init__()
            self.fc1 = nn.Linear(input_layer, 256)
            self.fc2 = nn.Linear(256, output_layer)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Hyperparameters
    input_size = 784
    output_size = 15
    learning_rate = 0.001
    batch_size = 128
    num_epoch = 8

    # Create DNN-instance
    model = DNN(input_size, output_size)
    # Loss function
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # we want to fully iterate the loader multiple times, called epochs, to train successfully
    for epoch in range(num_epoch):
        # here we fully iterate the loader each time
        for i, data in enumerate(trainloader):
            i = i  # i is just a counter you may use for logging purposes or such
            img, label = data  # data is a batch of samples, split into an image tensor and label tensor

            # As you may notice, img is of shape n x 1 x height x width, which means a batch of n matrices.
            # But fully connected neural network layers are designed to process vectors. You need to take care of that!
            # Also libraries like matplotlib usually expect images to be of shape height x width x channels.
            img = img.reshape(img.shape[0], -1)

            # forward propagation
            scores = model(img.float())
            loss = loss_func(scores, label)

            # zero previous gradients
            optimizer.zero_grad()

            # back-propagation
            loss.backward()

            # adam-optimizer's step
            optimizer.step()

    def checkAccuracy(loader, model):
        num_correct = 0
        num_sample = 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x = x.reshape(x.shape[0], -1)
                scores = model(x.float())
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_sample += preds.size(0)

                print("Total scanned samples: ", num_sample)
                print("Correct samples: ", num_correct)
                print("Accuracy: ", float(num_correct) / float(num_sample))
                print("----------------------------------------------------------")

        model.train()

    # checkAccuracy(trainloader, model)

    def accuracy(labels, preds):
        cm = confusion_matrix(labels, preds)
        df_cm = DataFrame(cm)
        sn.heatmap(df_cm, cmap='Oranges', annot=True)
        plt.show()
        

    # The code above is just given as a hint, you may change or adapt it.
    # Nevertheless, you are recommended to use the above loader with some batch size of choice.

    # Finally you have to submit, beneath the report, a csv file of predictions for the test data.
    # Extract the testdata using the provided method:
    # TODO
    # Use the network to get predictions (should be of shape 1500 x 15) and export it to csv using e.g. np.savetxt().

    # If you encounter problems during this task, please do not hesitate to ask for help!
    # Please check beforehand if you have installed all necessary packages found in requirements.txt
