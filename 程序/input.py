import xlrd
import xlwt
import numpy as np
from scipy.stats import pearsonr
# from keras import models, layers

# 读取数据
workbook = xlrd.open_workbook('F:/pycharm/attention/数据/CHP.xls')
sheet1_name = workbook.sheet_names()[0]
sheet1 = workbook.sheet_by_name('Data_CHP')
# row = sheet1.row_values(1)
# cols = sheet1.col_values(1)

# print(row[1:8])
# print(cols)

# 输入数据为（8776，7）
# 将excel的数据保存到data中
data = np.mat(np.zeros((8776, 7)))
for n in np.arange(1, 8777):
    data[n-1, :] = sheet1.row_values(n)[1:8]

# print(type(data))
# print(data.shape)

data = np.array(data[:, range(5)])
# 数据标准化
# mean = data.mean(axis=0)
# print("mean: ", mean)
# data -= mean
# std = data.std(axis=0)
# print("std: ", std)
# data /= std


def customMinMaxScaler(X):  # 最大最小标准化
    return (X - X.min()) / (X.max() - X.min())


for i in range(5):
    data[:, i] = customMinMaxScaler(data[:, i])

f = xlwt.Workbook()
sheet1 = f.add_sheet('0')
for j in range(5):
    for i in range(8776):
        sheet1.write(i, j, float(data[i, j]))
f.save(r'F:/pycharm/attention/数据/数据预处理.xls')
