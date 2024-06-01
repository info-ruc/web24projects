import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
hidden_dropout_prob = 0.3
num_labels = 4
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 1
batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "../dataset/"  
train_data_path = data_path + "train/train.csv"  # 训练数据集
valid_data_path = data_path + "test/test.csv"  # 验证数据集
human_value = 1
machine_value = 0
