import pandas as pd
import os
import torch
# self.data_dir = self.data_dir or default_data_path / self._name_
# get the absolute path of the .data.csv file inside this folder
file_path = "data.csv"
data_dir = os.path.abspath(__file__)
# print(self.data_dir)
data_dir = os.path.join(os.path.dirname(data_dir), file_path)
# print(self.data_dir)


# read data from csv file
data = pd.read_csv(data_dir)
# keep certain rows of data with "host" column == "intel-108"
data = data[data['host'] == 'intel-108']
data = torch.tensor(data['_value'].values, dtype=torch.float32)

print(torch.mean(data))