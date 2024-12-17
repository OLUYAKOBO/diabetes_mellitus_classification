import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

models = {"Logistic Regression": LogisticRegression(),
         "Random Forest": RandomForestClassifier(),
         'AdaBoost':AdaBoostClassifier(),
         'BaggingClassifier':BaggingClassifier(),
         'GradientBoost':GradientBoostingClassifier(),
         'Catboost':CatBoostClassifier(),
         'XGBoost':XGBClassifier()}

def fit_and_score(x_train,x_test,y_train,y_test):
    np.random.seed(42)
    model_scores = {}
    
    for name,model in models.items():
        #fit the train data to the models
        model.fit(x_train,y_train)
        
        #evaluate the score on the test data
        model_scores[name] = model.score(x_test,y_test)
        
    return model_scores