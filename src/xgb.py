import xgboost as xgb
import pandas as pd

class XGBoostModel:
    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {
                'max_depth': 3,
                'eta': 0.1,
                'objective': 'reg:squarederror'  # Adjust objective based on task
            }
        else:
            self.params = params

    def fit(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain)

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        return y_pred