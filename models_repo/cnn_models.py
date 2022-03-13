import os
import torch
import torch.nn as nn
from tqdm import tqdm, tqdm_notebook


def conv_block(in_channels, out_channels, batch_norm=True):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                          nn.ReLU(),
                          nn.BatchNorm2d(out_channels))
    return block


class Simple_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, 32),
                                      nn.MaxPool2d(2),
                                      conv_block(32, 64),
                                      conv_block(64, 64),
                                      nn.MaxPool2d(2),
                                      conv_block(64, 128),
                                      conv_block(128, 128),
                                      nn.AdaptiveAvgPool2d(7))
        self.classifier = nn.Sequential(nn.Linear(7 * 7 * 128, 1024),
                                        nn.Dropout(),
                                        nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x 

def simple_cnn(**kwargs):
    return Simple_CNN(**kwargs)