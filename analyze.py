# Import models

import pandas as pd
import numpy as np
import os

import torch
from torch import nn

import tqdm

import sys

import tarfile

import matplotlib
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(122 * 151, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork()
model.load_state_dict(torch.load('model_weights.pth'))

arguments = sys.argv

if len(arguments) != 2:
    print("Invalid arguments.")
    sys.exit()

current_dir = os.getcwd()
bin_file = os.path.join(current_dir, "bin")
correct_file = os.path.join(bin_file, "0")
incorrect_file = os.path.join(bin_file, "1")

if not os.path.exists(bin_file):
    os.mkdir(bin_file)
if not os.path.exists(correct_file):
    os.mkdir(correct_file)
if not os.path.exists(incorrect_file):
    os.mkdir(incorrect_file)

with tarfile.open(arguments[1]) as tar:

    for member in tar.getmembers():

        # Read Data
        f = tar.extractfile(member)
        byteData = f.read()
        data = np.fromstring(byteData.decode(), sep =' ')
        file_name = member.name.split('/')[-1][12:20]

        # Feed to neural network
        ml_data = torch.tensor(data[2::3]).float()
        raw_result = model(ml_data).item()
        result = round(raw_result)
        result_path = str(result)

        # Plot Data
        data = np.reshape(data, (18422, 3))
        data = np.swapaxes(data, 0, 1)
        x = data[0].reshape((122, 151))
        y = data[1].reshape((122, 151))
        z = data[2].reshape((122, 151))

        fig, ax = plt.subplots()

        rainbow = matplotlib.colors.LinearSegmentedColormap.from_list(name="rainbow", colors=['white', 'magenta', 'blue', 'cyan', 'lime', 'yellow', 'orange', 'red'])
        c = ax.pcolormesh(x, y, z, cmap=rainbow, vmin=0, vmax=0.30)
        ax.set_title(file_name)
        ax.set_xlabel(round(raw_result, 3))
        ax.axis([-1.69897, 2.854109, -200, -50])
        fig.colorbar(c, ax=ax)
        
        plt.savefig(f"bin/{result_path}/{file_name}.png")
        plt.close()