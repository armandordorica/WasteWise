import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, file_path, input_col, output_col):
        self.file_path = file_path
        self.input_col = input_col
        self.output_col = output_col
    
    def load_data(self):
        df = pd.read_csv(self.file_path)
        df[self.input_col] = df[self.input_col].astype(str).fillna('missing')
        df[self.output_col] = df[self.output_col].astype(str).fillna('missing')
        return df
    
    def split_data(self, df):
        train_indices, val_indices = train_test_split(range(len(df)), test_size=0.2)
        return train_indices, val_indices
