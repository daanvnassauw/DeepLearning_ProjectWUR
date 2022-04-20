#Import statements:
import Read_data
import glob
from sys import argv
from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import LogSoftmax
from torch.nn import Embedding
from torch import flatten


"""
Out channels kleiner maken.

Kernel size van de pooling layer aanpassen.
"""
#the network:
class LeNet(Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(LeNet, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv1d(in_channels=numChannels, out_channels=3,
            kernel_size=5)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=4, stride=1)

# Linear layer
        self.fc2 = Linear(in_features=100, out_features=1)
#initialize our softmax classifier
        self.logSoftmax = LogSoftmax(dim=1)


# Dropout toevoegen



#TIP: Netwerk kleiner maken en dropout toevoegen.


#Forward function:
    def forward(self, x):
        x = x.float()
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # through our only set of FC => RELU layers
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output
