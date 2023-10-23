from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


horizons = [1,5,10,20,30,50,70,80,100,110,120,130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,240, 250, 260, 270, 280, 290]
trained_pred_power = [1.78, 0.488, 0.405, 0.437, 0.474, 0.262, 0.407, 0.316, 0.453, 0.443, 0.535, 0.646, 0.309, 0.654, 0.767, 0.751, 0.716, 1.06, 1.121, 0.925, 1.142, 1.11, 1.177, 1.018, 0.884, 0.86, 1.237, 1.792]
trained_pred_power = np.array(trained_pred_power)
horizons = np.array(horizons)
plt.plot(horizons, trained_pred_power, label='test/loss')
plt.scatter(horizons, trained_pred_power, color='red', marker='o', label='loss_value')
plt.xlabel('horizons/step')
plt.xticks(np.arange(0, 300, 10))

plt.ylabel('power/W')
plt.title('test/loss')
plt.legend()
plt.show()
