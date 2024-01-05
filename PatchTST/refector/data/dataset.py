from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
from refector.data.timefeatures import time_features
import torch

class WeatherDataset(Dataset):
    def __init__(self, data_path, sequence_length, predict_length, 
                 label_length, scale=True, features='S', timeenc=0, 
                 use_time_features=False, split='train'):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.label_length = label_length
        self.predict_length = predict_length
        self.scale = scale
        self.features = features
        self.use_time_features =  use_time_features
        self.timeenc = timeenc

        assert split in ["train", "valid", "test"]
        type_map = {"train": 0, "valid": 1, "test": 2}
        self.set_type = type_map[split]

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 0, 0]
        border2s = [0, 0, 0]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]


        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
        elif self.features == 'S':
            cols_data = df_raw.columns[-1:]
        df_data = df_raw[cols_data]


        if self.scale:
            train_data = df_data[border1s[0]: border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values


        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = time_features(df_stamp, self.timeenc)


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.sequence_length
        r_begin = s_end - self.label_length
        r_end = r_begin + self.label_length + self.predict_length

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.use_time_features: return _torch(seq_x, seq_y, seq_x_mark, seq_y_mark)
        else: return _torch(seq_x, seq_y)


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def _torch(*dfs):
    return tuple(torch.from_numpy(x).float() for x in dfs)