# imports a getter for the StrangeSymbol Dataset loader and the test data tensor
from dataset import get_strange_symbols_train_loader, get_strange_symbols_test_data

if __name__ == '__main__':
    # executing this prepares a loader, which you can iterate to access the data
    trainloader = get_strange_symbols_train_loader(batch_size=128)

    # we want to fully iterate the loader multiple times, called epochs, to train successfully
    for epoch in range(0):
        # here we fully iterate the loader each time
        for i, data in enumerate(trainloader):
            i = i  # i is just a counter you may use for logging purposes or such
            img, label = data  # data is a batch of samples, split into an image tensor and label tensor

            # As you may notice, img is of shape n x 1 x height x width, which means a batch of n matrices.
            # But fully connected neural network layers are designed to process vectors. You need to take care of that!
            # Also libraries like matplotlib usually expect images to be of shape height x width x channels.
            print(img.shape)

    # TODO
    # Now it's up to you to define the network and use the data to train it.
    # The code above is just given as a hint, you may change or adapt it.
    # Nevertheless, you are recommended to use the above loader with some batch size of choice.

    # Finally you have to submit, beneath the report, a csv file of predictions for the test data.
    # Extract the testdata using the provided method:
    # TODO
    # Use the network to get predictions (should be of shape 1500 x 15) and export it to csv using e.g. np.savetxt().

    # If you encounter problems during this task, please do not hesitate to ask for help!
    # Please check beforehand if you have installed all necessary packages found in requirements.txt
