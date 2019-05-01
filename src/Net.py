import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, width, height, number_of_color_channels, number_of_first_convolution_output_channels, number_of_second_convolution_output_channels, number_of_fully_connected_nodes):
        super(Net, self).__init__()
        # new_width = (old_width - KERNEL + 2 * PADDING) / STRIDE + 1
        self.width=width
        self.height=height
        self.num_second_convolution_output_channels=number_of_second_convolution_output_channels

        # nn.Conv2d(num of input channels, num of output channels, kernel size - int or tuple, stride)
        self.conv1 = nn.Conv2d(
            number_of_color_channels,
            number_of_first_convolution_output_channels,
            kernel_size=(5,5), stride=1, padding=2
        )
        self.conv2 = nn.Conv2d(
            number_of_first_convolution_output_channels,
            number_of_second_convolution_output_channels,
            kernel_size=(5,5), stride=1, padding=2
        )
        self.conv2_bn = nn.BatchNorm2d(number_of_second_convolution_output_channels)

        # nn.Linear(size of input sample, size of output sample)
        self.fc1 = nn.Linear(
            int(width / 4 * height / 4 * number_of_second_convolution_output_channels),
            number_of_fully_connected_nodes
        )
        self.fc1_bn = nn.BatchNorm1d(number_of_fully_connected_nodes)
        self.fc2 = nn.Linear(number_of_fully_connected_nodes, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
        x = x.view(-1, int(self.width / 4 * self.height / 4 * self.num_second_convolution_output_channels))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)
        return torch.squeeze(torch.sigmoid(x))
