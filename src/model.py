"""sample"""
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def build_model(preprocess):
    """sample"""
    model = Pipeline([('preprocess',preprocess),('model',XGBClassifier())])
    return model

def train_model(model,x_train,y_train):
    """sample"""
    model.fit(x_train,y_train)

def predict(model,x_test):
    """sample"""
    pred = model.predict(x_test)
    return pred
