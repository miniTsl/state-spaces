import os
import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from src.dataloaders.base import SequenceDataset


class ServerTrain(Dataset):
    def __init__(self, train_x, train_y, forecast_horizon):
        self.x = train_x
        self.y = train_y
        self.forecast_horizon = forecast_horizon

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class ServerTest(Dataset):
    def __init__(self, test_x, test_y, forecast_horizon):
        self.x = test_x
        self.y = test_y
        self.forecast_horizon = forecast_horizon

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

class Server(SequenceDataset):

    _name_ = "server"
    d_input = 1
    d_output = 1


    # 注册新的参数，在dataset.yaml文件中指定具体的值
    @property
    def init_defaults(self):
        return {
            "permute": False,
            "val_split": 0.2,
            "seed": 42,  # For train/val split
            "forecast_horizon": 1,
            "prediction_window": 10
        }

    @property
    def l_output(self):
        return self.forecast_horizon

    @property
    def L(self):
        return 200
    
    def setup(self, ratio=0.9):
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

        # get data pair from data in the size of (prediction_window + forecast_horizon) into a list data_pair
        # self.L = self.prediction_window
        data_pair = []
        # print("self.prediction_windows: ", self.prediction_window)
        # print("self.forecast_hozizon: ", self.forecast_horizon)
        for i in range(len(data) - self.prediction_window - self.forecast_horizon + 1):
            data_pair.append(data[i:i + self.prediction_window + self.forecast_horizon].unsqueeze(-1))

        # shuffle the data_pair into two lists: train_data and test_data
        random.shuffle(data_pair)
        train_data = data_pair[:int(len(data_pair) * ratio)]
        test_data = data_pair[int(len(data_pair) * ratio):]
        # print(len(train_data), len(test_data))

        # seperate train_data into train_x and train_y
        train_x = [i[0:self.prediction_window] for i in train_data]
        train_y = [i[self.prediction_window:] for i in train_data]

        # seperate test_data into test_x and test_y
        test_x = [i[0:self.prediction_window]for i in test_data]
        test_y = [i[self.prediction_window:] for i in test_data]

        self.dataset_train = ServerTrain(train_x, train_y, self.forecast_horizon)
        self.dataset_test = ServerTest(test_x, test_y, self.forecast_horizon)

        self.split_train_val(self.val_split)

        # 针对forecasting的task，这里需要再加上吧forecast_horizon这个属性
        self.dataset_train.forecast_horizon = self.forecast_horizon
        self.dataset_test.forecast_horizon = self.forecast_horizon
        self.dataset_val.forecast_horizon = self.forecast_horizon


    def __str__(self):
        return f"{'p' if self.permute else 's'}{self._name_}"
    


if __name__ == "__main__":

    server = Server(_name_="Server")
    prediction_window = 2
    forecast_horizon = 1
    ratio = 0.9
    server.setup(prediction_window, forecast_horizon, ratio)

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