# import torch
# file_path = "outputs/2023-10-15/14-21-48-304097/checkpoints/val/loss.ckpt"

# loss_state = torch.load(file_path)
# print(loss_state)

# # 打开一个文件以供写入
# with open("loss.txt", "w") as file:
#     # 使用 print 函数将内容输出到文件
#     print(loss_state, file=file)
# # 文件会在退出上下文管理器后自动关闭

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 读取数据
data_prefix = "input200/forecasting/"
test_loss = []
forecast_horizon = [1,5,10,20,30,50,70,100,120,140,150,160,170,180,190]
for i in forecast_horizon:
    file_path = data_prefix + str(i) + ".txt"
    with open(file_path, "r") as f:
        data = f.readlines()[-500:]
        # 提取data中test/loss:之后，train/loss之前的数字
        data = [float(x.split("test/loss:")[1].split("train/loss:")[0]) for x in data]
        # 求最小的10个loss的平均值
        test_loss.append(sum(sorted(data)[:10]) / 10)

# plot
plt.figure(figsize=(10, 6))
plt.plot(forecast_horizon, test_loss, color='blue', label='test_loss')
plt.scatter(forecast_horizon, test_loss, color='red', marker='o', label='loss_value')
plt.xticks(np.arange(0, 200, 10))
plt.yticks(np.arange(0, 14, 1))
plt.xlabel('forecast_horizon')
plt.ylabel('test_loss(mae)')
plt.legend()
plt.title('test_loss(mae), prediction_window=200, max_epoch=500')
plt.show()

