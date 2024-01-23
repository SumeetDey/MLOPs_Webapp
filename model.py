import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature
from analytics.categorical_to_numerical import CategoricalColumnEncoder
from analytics.datasplitter import DataSplitter

df = pd.read_csv(r"/Users/deys/Desktop/Shasmita/HR_Analytics.csv.csv")
encoder = CategoricalColumnEncoder(df)
df=encoder.fit_transform()
splitter = DataSplitter(dataframe=df, target_column='Attrition', test_size=0.4, random_state=42)
X_train = splitter.X_train
X_test = splitter.X_test
y_train = splitter.y_train
y_test = splitter.y_test



##############mlflow will start############

mlflow.set_experiment("/mlops_project_end_to_end_randomforestclassifier")
mlflow.start_run()

model=RandomForestClassifier(n_estimators=200,min_samples_split=15, min_samples_leaf=8,random_state=42)

model.fit(X_train,y_train)

predictions=model.predict(X_test)
X_test["Actual"]=y_test
X_test["prediction"]=predictions
X_test.to_csv(r"E:\MLOPs_Webapp\prediction.csv",index=False)

signature=infer_signature(X_test,predictions)
accuracy=accuracy_score(y_test,predictions)


mlflow.log_metric("accuracy",accuracy)
mlflow.log_param("n_estimators",200)
mlflow.log_param("min_samples_split",15)
mlflow.log_param("min_samples_leaf",8)
mlflow.log_param("random_state",42)
mlflow.sklearn.log_model(model,"random_forest_model",signature=signature)
