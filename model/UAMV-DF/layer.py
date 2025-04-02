import numpy as np 

class layer(object):
    def __init__(self,layer_id):
        self.layer_id=layer_id
        self.estimators=[]
    
    def add_est(self,estimator):
        if estimator!=None:
            self.estimators.append(estimator)

    def predict_proba(self,x):
        proba=None
        for est in self.estimators:
            proba=est.predict_proba(x) if proba is None else np.hstack((proba,est.predict_proba(x)))
        return proba
    
    def _predict_proba(self,x_test):
        proba=None
        for est in self.estimators:
            proba=est.predict_proba(x_test) if proba is None else proba+est.predict_proba(x_test)
        proba/=len(self.estimators)
        return proba