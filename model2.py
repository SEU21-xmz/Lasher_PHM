import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def dataloader():
    base_dir = r"F:\桌面\DATA\DATA"
    dataset = []
    max_time = 2500
    for file_name in os.listdir(base_dir):
        if "Data" not in file_name:
            continue
        file = open(os.path.join(base_dir, file_name))
        least_time = 0
        least_rate = 0
        data = []
        while True:
            # 当前数据大于max_time时，停止读取
            if least_time >= max_time:
                break
            line = file.readline()
            if not line:
                break
            if line == "" or line == "\n":
                break
            # 固定读取前2500小时的数据作为数据集，模型输入维度是2000/5=400
            # 400维度转化为50*80
            time_now, result = line.split(",")
            time_now = int(round(float(time_now), 0))
            time_now = 0 if time_now < 0 else time_now
            # 对数据处理，如果下一个时间段是五的倍数,并且大于五的倍数,令其为5n+5
            if time_now / 5 > time_now // 5:
                time_now = time_now // 5 * 5 + 5
            if least_time == 0:
                # 刚开始进行数据填充
                # 初始时候第一条数据不为5时，每隔五个时间步填充一条数据，直到大于第一个时刻
                for i in range(time_now // 5):
                    data.append(float(result))
                    least_rate = float(result)
                    least_time = time_now
                continue
            # 下一时刻的时间小于上一时刻的时间，证明出现了数据异常，停止读取
            if float(time_now) < least_time:
                continue
            if time_now - least_time >= 5:
                flag = True
                for i in range((time_now - least_time) // 5):
                    # 将此条数据和上一条数据做差值，得到时间跨度
                    # 并根据时间跨度，将数据跨度均匀等分,并填充
                    data.append(least_rate + i * (float(result) - least_rate) // (time_now - least_time // 5))
                    if least_time + i * 5 >= max_time:
                        flag = False
                        least_time = max_time
                        least_rate = float(result)
                        break
                if flag:
                    least_time = time_now
                    least_rate = float(result)
        # 不足max时填充数据,填充值为数组最后一个数值
        # if len(data) < max_time // 5:
        #     data = np.pad(data, (0, max_time // 5 - len(data)), constant_values=data[-1])
        """
            下面的操作是对数据进行切片，使每一条数据的长度都一致
        """
        if len(dataset) == 0:
            dataset.append(data)
        else:
            if len(dataset[0]) < len(data):
                dataset.append(data[:len(dataset[0])])
            else:
                dataset = np.array(dataset)
                dataset = dataset[:, :len(data)]
                dataset = list(dataset)
                dataset.append(data)
    return np.array(dataset)


class LSTMModel(nn.Module):
    # 定义长短期记忆网络
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
         self.hidden = nn.Sequential(
             nn.LSTM(hidden_dim, 128, batch_first=True),
             nn.LSTM(128, 256, batch_first=True),
             nn.LSTM(256, 512, batch_first=True),
             nn.LSTM(512, 256, batch_first=True),
             nn.LSTM(256, 128, batch_first=True),
             nn.LSTM(128, 64, batch_first=True),
             nn.LSTM(64, 8, batch_first=True),
         )
        self.fc = nn.Linear(hidden_dim, output_dim)

    # 前向传播
    def forward(self, x):
        out, _ = self.lstm(x)
        # out = self.hidden(out)
        out = self.fc(out[:, -1, :])
        return out


def train(datasets):
    # 超参数设置
    input_dim = 1  # 输入维度（光功率）
    hidden_dim = 64  # LSTM隐藏层维度
    output_dim = 1  # 输出维度（预测的光功率）
    num_epochs = 1000  # 训练轮数
    learning_rate = 0.001  # 学习率
    # 将数据进行切割划分
    train_x, test_x = train_test_split(datasets, test_size=0.2, random_state=21)
    # 转化为torch可以使用的Tensor类型
    train_X = torch.from_numpy(train_x).float()
    test_X = torch.from_numpy(test_x).float()

    # 创建模型实例
    model = LSTMModel(input_dim, hidden_dim, output_dim)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 模型训练
    loss_list = []
    for epoch in range(num_epochs):
        loss_sum = 0
        for train_data in train_X:
            if min(train_data > 90):
                continue
            for i in range(1, len(train_data)):
                # 将当前时间步及之前的每条数据作为数据输入模型
                input_sequence_tensor = torch.FloatTensor(train_data[:i]).unsqueeze(0).unsqueeze(2)
                model.train()
                # 模型得到下一时刻的预测值
                outputs = model(input_sequence_tensor)
                # 预测值和真实值求损失
                loss = criterion(outputs, train_data[i])

                # loss = criterion(outputs, labels) 的意思是计算模型输出和标签之间的损失。
                # 其中，outputs 是模型的输出，labels 是标签，criterion 是损失函数。
                # 计算出的损失值可以用来优化模型的参数，使得模型的输出更加接近标签。

                # print(loss)
                optimizer.zero_grad()
                # optimizer.zero_grad() 是 PyTorch
                # 中定义优化器的一个方法，它会将模型中所有可训练的参数的梯度清零。
                # 在训练神经网络时，通常需要在每次迭代之前调用这个函数。因为如果不清零梯度，那么优化器在更新权重时会累加之前的梯度。
                # 反向传播，更新参数
                loss.backward()
                optimizer.step()
                # 和 optimizer.zero_grad() 配合使用，用于实现梯度下降算法中的参数更新步骤。
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
                loss_sum += loss.item()
        loss_list.append(loss_sum / len(train_X))
    plt.plot(range(len(loss_list)), loss_list, label="loss")
    plt.savefig("loss2.png")
    torch.save(model.state_dict(), "model2.pth")


# 进行光功率序列的预测


def eval_model():
    """
        模型验证类似于训练过程不过多了一条加载模型权重部分
        其他类似train
        model.load_state_dict(torch.load('model2.pth'))
    :return:
    """
    datasets = dataloader()
    train_x, input_sequence = train_test_split(datasets, test_size=0.2, random_state=21)
    input_sequence = torch.from_numpy(input_sequence).float()
    train_x = torch.from_numpy(train_x).float()
    model = LSTMModel(1, 64, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.load_state_dict(torch.load('model2.pth'))
    model.eval()
    th = 80
    # with torch.no_grad():
    predict_time = True
    true_time = True
    criterion = nn.MSELoss()
    for input_value in input_sequence:
        true_label = []
        predict_label = []
        loss_a = []
        for i in range(1, len(input_value)):
            input_sequence_tensor = torch.FloatTensor(input_value[:i]).unsqueeze(0).unsqueeze(2)
            outputs = model(input_sequence_tensor)
            predict_label.append(outputs.item())
            true_label.append(input_value[i].item())
            loss = criterion(outputs, input_value[i])
            loss_a.append(loss.item())
            # print(f"误差:{loss}")
            # print(input_value[i])
            optimizer.zero_grad()
            # loss.backward()  # 模型不进行反向传播更新参数(防止数据对训练结果造成影响)
            optimizer.step()
            # print(outputs)
            if predict_time and outputs < th:
                # i是时间步,所以 * 5是真实时间
                print(f"预警时间为{i * 5},{outputs}")
                predict_time = False
            if true_time and input_value[i] < th:
                print(f"真实时间为{i * 5}")
                true_time = False
        plt.scatter(predict_label, true_label)
        plt.savefig("./a.png")
        plt.clf()
        plt.plot(range(len(loss_a)), loss_a)
        plt.savefig("./b.png")
        plt.clf()


if __name__ == '__main__':  # 判断当前模块是否是主程序入口。如果当前模块是主程序入口，则执行该语句块中的代码；如果当前模块是被其他模块导入的，则不执行该语句块中的代码。
    # data_set = dataloader()
    # train(data_set)
    eval_model()
