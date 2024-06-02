from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from icecream import ic 
from config import train_data_path,  valid_data_path, batch_size


class TextDataset(Dataset):
    def __init__(self, path_to_file):
        self.dataset = []
        df = pd.read_csv(path_to_file, header = None)
        for index, row in df.iterrows():
            self.dataset.append({'label': row[0] - 1, 'text': row[2]})

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 根据 idx 分别找到 text 和 label
        text = self.dataset[idx]['text']
        label = self.dataset[idx]['label']
        sample = {"text": text, "label": label}
        # 返回一个 dict
        return sample

# 加载训练集
text_train_set = TextDataset(train_data_path)
text_train_loader = DataLoader(text_train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# 加载验证集
text_valid_set = TextDataset(valid_data_path)
text_valid_loader = DataLoader(text_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
