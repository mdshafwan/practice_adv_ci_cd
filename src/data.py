"""sample"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
def load_dataset():
    """sample"""
    df = pd.read_csv("src/data/customer_churn_dataset-testing-master.csv")
    return df
def preprocess(df):
    """sample"""
    x = df.drop('Churn',axis=1)
    y = df['Churn']
    num = x.select_dtypes(include=['int64','float64']).columns.tolist()
    cat = x.select_dtypes(include=['str','object']).columns.tolist()
    num_pipe = Pipeline([('impute',SimpleImputer(strategy='median')),
                         ('scale',StandardScaler())])
    cat_pipe = Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                         ('scale',OneHotEncoder(handle_unknown='ignore',drop='first'))])
    preproces = ColumnTransformer([('num',num_pipe,num),
                                   ('cat',cat_pipe,cat)])
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.2)
    return x_train,x_test,y_train,y_test,preproces
