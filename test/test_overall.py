from src.model import build_model, predict, train_model
from src.metric import evalution
from sklearn.pipeline import Pipeline
class Testing():
    def test_overall(self,pre):
        x_train,x_test,y_train,y_test,preprocess = pre
        model = build_model(preprocess)
        train_model(model,x_train,y_train)
        pred = predict(model,x_test)
        eval = evalution(y_test,pred)
        assert len(pre) == 5 and isinstance(build_model(preprocess),Pipeline) and eval['acc'] > 0.5
