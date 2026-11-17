import pytest
from src.data import load_dataset,preprocess
from src.model import build_model,train_model,predict

@pytest.fixture
def raw():
    return load_dataset()

@pytest.fixture
def pre(raw):
    return preprocess(raw)

@pytest.fixture
def model(pre):
    x_train,x_test,y_train,y_test,preproces = pre
    model = build_model(preproces)
    train_model(model,x_train,y_train)
    pred = predict(model,x_test)
    return pred
