import pandas as pd

class CSVDataLoader:
    def __init__(self, file_path, batch_size=None):
        self.file_path = file_path
        self.batch_size = batch_size
        self.data = pd.read_csv(file_path)
        self.num_samples = len(self.data)
        self.current_idx = 0

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.batch_size is None:
            if self.current_idx == 0:
                self.current_idx = self.num_samples
                return self.data
            else:
                raise StopIteration
        else:
            if self.current_idx >= self.num_samples:
                raise StopIteration
            batch = self.data.iloc[self.current_idx:self.current_idx+self.batch_size]
            self.current_idx += self.batch_size
            return batch

