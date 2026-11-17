import os
from src.model import build_model
from sklearn.pipeline import Pipeline

class Test():
    def test_preprocess_len(self,pre):
        assert len(pre) == 5

    def test_data_loc(self):
        assert os.path.exists('src/data/customer_churn_dataset-testing-master.csv')

    def test_model(self,pre):
        *_,preprocess = pre
        assert isinstance(build_model(preprocess),Pipeline)

    