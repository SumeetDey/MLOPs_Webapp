from sklearn.model_selection import train_test_split
import pandas as pd

class DataSplitter:
    def __init__(self, dataframe, target_column, test_size=0.4, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            dataframe.drop(target_column, axis=1),
            dataframe[target_column],
            test_size=test_size,
            random_state=random_state
        )


