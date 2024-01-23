import pandas as pd
from sklearn.preprocessing import LabelEncoder

class CategoricalColumnEncoder:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.label_encoder = LabelEncoder()
        self.categorical_columns = None

    def fit_transform(self):
        self.categorical_columns = self.dataframe.select_dtypes(include=['object']).columns

        for col in self.categorical_columns:
            self.dataframe[col] = self.label_encoder.fit_transform(self.dataframe[col])
        return self.dataframe




