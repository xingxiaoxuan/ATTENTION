import xlrd
import xlwt
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE

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

real_data = np.array(data[:, range(5)])
data = np.array(data[:, range(5)])

# 数据标准化
# mean = data.mean(axis=0)
# data -= mean
# std = data.std(axis=0)
# data /= std
target = 0  # 0冷 1热 2电


# def return_back(data):
#     X = real_data[:, target]
#     print("X.min, X.max: ", X.min(), X.max())
#     data = data * (X.max() - X.min()) + X.min()
#     return data

X = real_data[:, target]
print("X.min, X.max: ", X.min(), X.max())


def return_back(dat):
    X = real_data[:, target]
    print("X.min, X.max: ", X.min(), X.max())
    for i in range(72):
        data[i] = dat[i] * (X.max() - X.min()) + X.min()
    return data


def customMinMaxScaler(X):  # 最大最小标准化
    return (X - X.min()) / (X.max() - X.min())


for i in range(5):
    data[:, i] = customMinMaxScaler(data[:, i])

series = data[:, target]  # 0冷 1热 2电
print(series)
print(len(series))  # 8776
size = int(len(series) * 0.66)
train, test = series[0:8704], series[8704:len(series)]
print(train)
# train, test = series[0:size], series[size:len(series)]
history = [x for x in train]
predictions = list()

# # fit model
# model = ARIMA(train, order=(5, 1, 0))
# model_fit = model.fit(disp=0)
# output = model_fit.forecast()
# # print("output: ", output[0])
#
# yhat = output[0]
# predictions.append(yhat)
#
# print(model_fit.summary())
# residuals = DataFrame(model_fit.resid)
# residuals.plot()
# plt.show()
# residuals.plot(kind='kde')
# plt.show()
# print(residuals.describe())

for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    # yhat = return_back(output[0])
    predictions.append(yhat)
    obs = test[t]
    # obs = return_back(test[t])
    history.append(obs)
    print('predicted = %f, expected = %f' % (yhat, obs))

# test = return_back(test)
# predictions = return_back(predictions)

MSE = mean_squared_error(test, predictions)
MAE = mean_absolute_error(test, predictions)
RMSE = np.sqrt(mean_squared_error(test, predictions))  # RMSE就是对MSE开方即可

print('Test MSE: %.3f' % MSE)
print('Test MAE: %.3f' % MAE)
print('Test RMSE: %.3f' % RMSE)
# 冷：Test MSE: 0.001
# Test MAE: 0.020
# Test RMSE: 0.028

# 热：Test MSE: 0.000
# Test MAE: 0.008
# Test RMSE: 0.011

# 电：Test MSE: 0.001
# Test MAE: 0.024
# Test RMSE: 0.033

# test = (test * std[target]) + mean[target]
# predictions = (np.array(predictions) * std[target]) + mean[target]

plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

f = xlwt.Workbook()
sheet1 = f.add_sheet('%d' % target)
for i in range(72):
    sheet1.write(i, 0, float(test[i]))
    sheet1.write(i, 1, float(predictions[i]))
f.save(r'F:/pycharm/attention/数据/ARIMA_test.xls')
