import xlrd
import xlwt
import numpy as np
from keras import Model
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from keras.callbacks import LearningRateScheduler
# from keras.callbacks import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.test.is_gpu_available())

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

# print(data)

data = np.array(data)
data = data[:, range(5)]

print(data.shape)

# X_min = []
# X_max = []
# for i in range(3):
#     x_min = data[:, i].min()
#     x_max = data[:, i].max()
#     X_min.append(x_min)
#     X_max.append(x_max)
# print("X_max: ", X_max)
# print("X_min: ", X_min)
# 数据归一化


def customMinMaxScaler(X):  # 最大最小标准化
    return (X - X.min()) / (X.max() - X.min())


for i in range(5):
    data[:, i] = customMinMaxScaler(data[:, i])

# 数据标准化
# mean = data[:6000].mean(axis=0)
# # print("mean: ", mean[2])
# data[:6000] -= mean
# std = data[:6000].std(axis=0)
# data[:6000] /= std


# 数据标准化
# mean = data[:6000].mean(axis=0)
# data -= mean
# std = data[:6000].std(axis=0)
# data /= std
# print(data)


def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = 0.001
    factor = 0.25
    dropEvery = 10
    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    # return the learning rate
    return float(alpha)


callbacks = [LearningRateScheduler(step_decay)]


def generator(data, target, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=1):
    """
    将要用到的生成器 他生成了一个元组（samples, targets）， 其中samples是输入数据的一个批量，targets是对应的目标温度数组。
    :param data: 浮点数数组组成的原始数组，已经标准化。
    :param target: 预测对象 0冷 1热 2电
    :param lookback: 输入数据应该包括过去多少个时间步。
    :param delay: 目标应该在未来多少个时间步之后。
    :param min_index: data数组中的索引，用于界定需要抽取哪些时间步。
    :param max_index: 保存一部分数据用于验证，另一部分用于测试。
    :param shuffle: 打乱样本，还是按顺序抽取样本。
    :param batch_size: 每个样本的批量数。
    :param step: 数据采样的周期（单位：时间步）。我们将其设置为1，为的是每小时抽取一个数据点。
    :return:
    """
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            # if i + batch_size >= max_index:
            #     i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

            # 初始化samples，targets
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), ))

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][target]  # 0冷 1热 2电
        yield samples, targets


lookback = 720  # 前三十天的数据
step = 1
delay = 2
batch_size = 72

target_num = 1  # 0冷 1热 2电................................................

train_gen = generator(data, target_num, lookback=lookback, delay=delay, min_index=0, max_index=6000,
                      shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(data, target_num, lookback=lookback, delay=delay, min_index=6001, max_index=7500,
                    step=step, batch_size=batch_size)
test_gen = generator(data, target_num, lookback=lookback, delay=delay, min_index=7501, max_index=None,
                     step=step, batch_size=batch_size)
# 查看，需要从 generate 中抽取多少次
val_steps = (7500 - 6001 - lookback) // batch_size
test_steps = (len(data) - 7501 - lookback) // batch_size


def get_base_model_history():
    result_72 = np.array(np.zeros((72, 1)))  # 预测最后3天的数据
    target_72 = np.array(np.zeros((72, 1)))  # 最后3天数据的真实值
    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // step, data.shape[-1])))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=10,
                                  validation_data=val_gen, validation_steps=val_steps)
    for steps in range(test_steps + 1):
        samples, targets = next(test_gen)
        preds = model.predict(samples)
        # P.append(preds)
        if steps == 4:
            print("preds.shape: ", preds.shape)  # (72, 1)
            print("targets.shape: ", targets.shape)  # (72,)
            print(preds)
            print(targets)
            result_72 = preds
            target_72 = targets.reshape(72, 1)
    print("result_72: ", result_72)
    print("target_72: ", target_72)

    # result_72 = (result_72 * (X_max[target_num] - X_min[target_num])) + X_min[target_num]  # 0冷 1热 2电
    # target_72 = (target_72 * (X_max[target_num] - X_min[target_num])) + X_min[target_num]  # 0冷 1热 2电

    MSE = mean_squared_error(target_72, result_72)
    MAE = mean_absolute_error(target_72, result_72)
    RMSE = np.sqrt(mean_squared_error(target_72, result_72))  # RMSE就是对MSE开方即可

    plt.plot(range(72), target_72, 'b', label='real')
    plt.plot(range(72), result_72, 'r', label='predict')
    plt.show()

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('epochs_30')
    for i in range(72):
        sheet1.write(i, 0, float(target_72[i]))
        sheet1.write(i, 1, float(result_72[i]))
    f.save(r'base_model.xls')

    print("MSE, MAE, RMSE: ", MSE, MAE, RMSE)

    return history


# 使用GRU 的模型

def get_gru_model_history():
    result_72 = np.array(np.zeros((72, 1)))  # 预测最后3天的数据
    target_72 = np.array(np.zeros((72, 1)))  # 最后3天数据的真实值
    model = Sequential()
    model.add(layers.GRU(16, dropout=0.5, input_shape=(None, data.shape[-1])))
    model.add(layers.BatchNormalization())
    # model.add(layers.GRU(32, dropout=0.5, recurrent_dropout=0.5,
    #                      return_sequences=True, input_shape=(None, data.shape[-1])))
    # model.add(layers.GRU(64, activation='relu', dropout=0.5, recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    # model.add(layers.BatchNormalization())

    opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='mae', metrics=['acc'])

    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')
    history = model.fit_generator(train_gen, steps_per_epoch=100, epochs=5, callbacks=callbacks,
                                  validation_data=val_gen, validation_steps=val_steps)

    for steps in range(test_steps + 1):
        samples, targets = next(test_gen)
        preds = model.predict(samples)
        # P.append(preds)
        if steps == 4:
            print("preds.shape: ", preds.shape)  # (72, 1)
            print("targets.shape: ", targets.shape)  # (72,)
            print(preds)
            print(targets)
            result_72 = preds
            target_72 = targets.reshape(72, 1)
    print("result_72: ", result_72)
    print("target_72: ", target_72)

    # result_72 = (result_72 * (X_max[target_num] - X_min[target_num])) + X_min[target_num]  # 0冷 1热 2电
    # target_72 = (target_72 * (X_max[target_num] - X_min[target_num])) + X_min[target_num]  # 0冷 1热 2电

    MSE = mean_squared_error(target_72, result_72)
    MAE = mean_absolute_error(target_72, result_72)
    RMSE = np.sqrt(mean_squared_error(target_72, result_72))  # RMSE就是对MSE开方即可

    plt.plot(range(72), target_72, 'b', label='real')
    plt.plot(range(72), result_72, 'r', label='predict')
    plt.show()

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('epochs_30')
    for i in range(72):
        sheet1.write(i, 0, float(target_72[i]))
        sheet1.write(i, 1, float(result_72[i]))
    f.save(r'GRU_model.xls')

    print("MSE, MAE, RMSE: ", MSE, MAE, RMSE)
    return history


# 使用 dropout 正则化的基于 GRU 的模型

def get_gru_model_with_dropout_history():

    model = Sequential()
    model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2,
                         input_shape=(None, data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                                  validation_data=val_gen, validation_steps=val_steps)
    model.save('gru_model_with_dropout.h5')
    return history


# 使用 dropout 正则化的堆叠 GRU 模型

def get_mul_gru_model_with_dropout_history():

    model = Sequential()
    model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5,
                         return_sequences=True, input_shape=(None, data.shape[-1])))
    model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                                  validation_data=val_gen, validation_steps=val_steps)
    model.save('mul_gru_model_with_dropout')
    return history


def get_LSTM_mdoel_history():
    model = Sequential()
    model.add(layers.LSTM(32, input_shape=(None, data.shape[-1])))
    model.add(layers.Dense(1))
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=10,
                                  validation_data=val_gen, validation_steps=val_steps)
    model.save('mul_LSTM_model')
    return history


def attention_lstm():
    input = layers.Input(shape=(lookback // step, data.shape[-1]))
    encoder = layers.LSTM(32, return_sequences=True)(input)

    # attention
    attention_pre = layers.Dense(1, name='activation_vec')(encoder)
    attention_porbs = layers.Softmax()(attention_pre)
    attention_mul = layers.Lambda(lambda x: x[0]*x[1])([attention_porbs, encoder])

    decoder = layers.LSTM(32, return_sequences=True)(attention_mul)
    output = layers.Flatten()(decoder)
    output = layers.Dense(1)(output)
    model = Model(inputs=input, outputs=output)
    model.summary()
    model.compile(optimizer=RMSprop(), loss='mae', metrics=['acc'])
    history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                                  validation_data=val_gen, validation_steps=val_steps)
    model.save('mul_LSTM_model')
    return history


def draw_loss(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


draw_loss(get_base_model_history())
# draw_loss(history=get_gru_model_history())
# draw_loss(history=get_gru_model_with_dropout_history())
# draw_loss(history=get_mul_gru_model_with_dropout_history())
# draw_loss(history=get_LSTM_mdoel_history())
# draw_loss(history=attention_lstm())

# get_base_model_history()
