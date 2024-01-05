from torch.utils.data import DataLoader
from dataset import WeatherDataset
from parameter import Parameter


class WeatherDataLoader:
    def __init__(self, data_path:str, batch_size:int, num_workers:int, 
                 shuffle_train:bool, shuffle_valid:bool, sequence_length:int, 
                 predict_length:int, label_length:int, scale:str, features:str, 
                 timeenc:int, use_time_features:int):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.predict_length = predict_length
        self.label_length = label_length
        self.scale = scale
        self.features = features
        self.timeenc = timeenc
        self.use_time_features = use_time_features
        
        self.train_dataloader = self._make_dataloader(
            split="train", 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle_train
        )

        self.valid_dataloader = self._make_dataloader(
            split="valid", 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=shuffle_valid
        )

        self.test_dataloader = self._make_dataloader(
            split="test", 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=False
        )

    def _make_dataloader(self, split, batch_size, num_workers, shuffle=False):
        dataset = WeatherDataset(
            data_path=self.data_path, 
            sequence_length=self.sequence_length, 
            predict_length=self.predict_length, 
            label_length=self.label_length,
            scale=self.scale,
            features=self.features,
            timeenc = self.timeenc,
            use_time_features=self.use_time_features,
            split=split
        )

        if len(dataset) == 0: return None

        return DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers
        )