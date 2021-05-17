import numpy as np
import xlrd
import xlwt
from minepy import MINE

# 读取数据
workbook = xlrd.open_workbook('F:/pycharm/attention/数据/CHP.xls')
sheet1_name = workbook.sheet_names()[0]
sheet1 = workbook.sheet_by_name('Data_CHP')

# 输入数据为（8776，7）
# 将excel的数据保存到data中
data = np.mat(np.zeros((8776, 7)))
for n in np.arange(1, 8777):
    data[n-1, :] = sheet1.row_values(n)[1:8]

# print(type(data))
# print(data.shape)

f = xlwt.Workbook()
sheet2 = f.add_sheet('sheet1')

data = np.array(data)
result = np.mat(np.zeros((7, 7)))
mine = MINE(alpha=0.6, c=15)

# mine.compute_score(data[:, 1], data[:, 3])
# a = mine.mic()
# print(mine.mic())

for i in range(7):
    for j in range(7):
        mine.compute_score(data[:, i], data[:, j])
        a = mine.mic()
        result[i, j] = a
        sheet2.write(i, j, a)
print(result)

# sheet2.write(0, 1, a)
f.save(r'test2.xls')
