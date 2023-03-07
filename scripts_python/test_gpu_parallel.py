"""
SEE HERE  https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

Conclusions:
    - DistributedDataParallel is supposedly faster than DataParallel.
    - DistributedDataParallel does not work with lists of tensor.
"""


from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Parameters
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Generate random tensors as data
class RandomDataset(Dataset):
    def __init__(self, nb_features, nb_data):
        self.len = nb_data
        self.data = torch.randn(nb_data, nb_features)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def copied_from_tutorial():
    # Model: simple linear model
    class Model(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, input: torch.Tensor):
            output = self.fc(input)
            print("\tIn Model: input size", input.size(),
                  "output size", output.size())

            return output

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                             batch_size=batch_size, shuffle=True)

    model = Model(input_size, output_size)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    model.to(device)

    # Run the model
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        print("Outside: input size", input.size(),
              "output_size", output.size())


def using_lists():
    """
    Conclusion: This does not work. Splits every tensor inside the list!

    To use parallel GPUs, we would need to concatenate data first...
    See fmassa comment here: https://github.com/pytorch/vision/issues/1659
    """
    # Model: simple linear model
    class ListModel(nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.fc = nn.Linear(input_size, output_size)

        def forward(self, input: List[torch.Tensor]):
            print("Concatenating list of {} elements of shape {}".format(len(input), input[0].size()))
            input = torch.vstack(input)

            output = self.fc(input)
            print("\tIn Model: input size", input.size(),
                  "output size", output.size())

            return output

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                             batch_size=batch_size, shuffle=True)

    model = ListModel(input_size, output_size)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    model.to(device)

    # Run the model
    for data in rand_loader:
        input = data.to(device)

        # Convert input to a list of inputs
        input = [input[i] for i in range(len(input))]
        output = model(input)
        print("Outside: input size {} x {}, output_size"
              .format(len(input), input[0].size(), output.size()))


def main():
    print("------RUNNING TUTORIAL")
    copied_from_tutorial()

    print("\n\n------USING LISTS")
    using_lists()


if __name__ == '__main__':
    main()
