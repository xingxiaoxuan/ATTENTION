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

result = np.mat(np.zeros((8000, 1)))

data = np.array(data)
data = np.squeeze(data)

f = xlwt.Workbook()
sheet1 = f.add_sheet('sheet1')

for i in range(8000):
    a = pearsonr(data[8000, :], data[i, :])[0]
    sheet1.write(i, 0, a)
    result[i, 0] = a

print(result)

f.save(r'test1.xls')
