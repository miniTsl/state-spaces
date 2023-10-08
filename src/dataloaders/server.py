import os
import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from src.dataloaders.base import SequenceDataset


class ServerTrain(Dataset):
    def __init__(self, train_x, train_y):
        self.x = train_x
        self.y = train_y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class ServerTest(Dataset):
    def __init__(self, test_x, test_y):
        self.x = test_x
        self.y = test_y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class Server(SequenceDataset):

    _name_ = "Server"
    # l_output = 0
    # L = 784

    @property
    def init_defaults(self):
        return {
            "permute": False,
            "val_split": 0.2,
            "seed": 42,  # For train/val split
        }

    def setup(self, history=2, horizon=1, ratio=0.9):
        # self.data_dir = self.data_dir or default_data_path / self._name_
        # get the absolute path of the .data.csv file inside this folder
        file_path = "data.csv"
        self.data_dir = os.path.abspath(__file__)
        # print(self.data_dir)
        self.data_dir = os.path.join(os.path.dirname(self.data_dir), file_path)
        # print(self.data_dir)


        # read data from csv file
        data = pd.read_csv(self.data_dir)
        # keep certain rows of data with "host" column == "intel-108"
        data = data[data['host'] == 'intel-108']
        data = torch.tensor(data['_value'].values, dtype=torch.float32)

        # get data pair from data in the size of (history + horizon) into a list data_pair
        self.d_input = history
        self.d_output = horizon
        data_pair = []

        for i in range(len(data) - history - horizon + 1):
            data_pair.append(data[i:i + history + horizon])

        # shuffle the data_pair into two lists: train_data and test_data
        random.shuffle(data_pair)
        train_data = data_pair[:int(len(data_pair) * ratio)]
        test_data = data_pair[int(len(data_pair) * ratio):]
        print(len(train_data), len(test_data))

        # seperate train_data into train_x and train_y
        train_x = [i[0:history] for i in train_data]
        train_y = [i[history:] for i in train_data]

        # seperate test_data into test_x and test_y
        test_x = [i[0:history] for i in test_data]
        test_y = [i[history:] for i in test_data]

        self.dataset_train = ServerTrain(train_x, train_y)
        self.dataset_test = ServerTest(test_x, test_y)

        # transform_list = [
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Lambda(lambda x: x.view(self.d_input, self.L).t()),
        # ]  # (L, d_input)
        # if self.permute:
        #     # below is another permutation that other works have used
        #     # permute = np.random.RandomState(92916)
        #     # permutation = torch.LongTensor(permute.permutation(784))
        #     permutation = permutations.bitreversal_permutation(self.L)
        #     transform_list.append(
        #         torchvision.transforms.Lambda(lambda x: x[permutation])
        #     )
        # TODO does MNIST need normalization?
        # torchvision.transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
        # transform = torchvision.transforms.Compose(transform_list)
        # self.dataset_train = torchvision.datasets.MNIST(
        #     self.data_dir,
        #     train=True,
        #     download=True,
        #     transform=transform,
        # )
        # self.dataset_test = torchvision.datasets.MNIST(
        #     self.data_dir,
        #     train=False,
        #     transform=transform,
        # )
        self.split_train_val(self.val_split)

    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"
    


if __name__ == "__main__":

    server = Server(_name_="Server")
    history = 2
    horizon = 1
    ratio = 0.9
    server.setup(history, horizon, ratio)

    print(server.data_dir)
    print(server.dataset_train)
    print(server.dataset_test)
    print(server.dataset_train[0])
    print(server.dataset_test[0])
    print(server.d_input)
    print(server.d_output)
    print(server.dataset_train[0][0].shape)
    print(server.dataset_train[0][1].shape)
    print(server.dataset_test[0][0].shape)
    print(server.dataset_test[0][1].shape)
    print(server.dataset_train[0][0])
    print(server.dataset_train[0][1])
    print(server.dataset_test[0][0])
    print(server.dataset_test[0][1])
    print(server.dataset_train[0][0].dtype)
    print(server.dataset_train[0][1].dtype)
    print(server.dataset_test[0][0].dtype)
    print(server.dataset_test[0][1].dtype)
    print(server.dataset_train[0][0].device)
    print(server.dataset_train[0][1].device)
    print(server.dataset_test[0][0].device)
    print(server.dataset_test[0][1].device)