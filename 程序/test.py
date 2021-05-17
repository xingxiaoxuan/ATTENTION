# 计算ARIMA的误差

import xlrd
import numpy as np
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE

# 读取数据
workbook = xlrd.open_workbook(r'F:/pycharm/attention/数据/ARIMA_test.xls')
sheet1_name = workbook.sheet_names()[0]
sheet1 = workbook.sheet_by_name('Sheet1')
# row = sheet1.row_values(1)
# cols = sheet1.col_values(1)

# print(row[1:8])
# print(cols)

# 输入数据为（8776，7）
# 将excel的数据保存到data中
data = np.mat(np.zeros((72, 6)))
for n in range(72):
    data[n, :] = sheet1.row_values(n)[:]

print(type(data))
print(data.shape)
print(np.arange(0, 6, 2))
# MSE = mean_squared_error(data[:, 1], data[:, 2])
# MAE = mean_absolute_error(data[:, 1], data[:, 2])
# RMSE = np.sqrt(mean_squared_error(data[:, 1], data[:, 2]))  # RMSE就是对MSE开方即可

for i in np.arange(0, 6, 2):
    MSE = mean_squared_error(data[:, i], data[:, i+1])
    MAE = mean_absolute_error(data[:, i], data[:, i+1])
    RMSE = np.sqrt(mean_squared_error(data[:, i], data[:, i+1]))  # RMSE就是对MSE开方即可]
    print("%d ：" % i)
    print('Test MSE: %.3f' % MSE)
    print('Test MAE: %.3f' % MAE)
    print('Test RMSE: %.3f' % RMSE)

if True:
    print(1)
else:
    print(2)
