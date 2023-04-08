import numpy as np
from scipy.optimize import minimize


class KernelLogisticRegression:

    def __init__(self, kernel, seed = 0, epsilon = 1e-6, **kernel_kwargs):
        self.__rs = np.random.RandomState(seed)
        self.epsilon = epsilon

        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs

        self.X_train = None
        self.v = None

    def fit(self, X, y) -> None:
        X = self.__get_biased_X(X)
        self.__fit(X, y)

    def __fit(self, X_, y) -> None:
        
        self.X_train = X_
        n_samples, _ = X_.shape

        self.v = self.__rs.normal(size = n_samples) # weights need to be centered around 0 or we will not reach the global minimum


        # for epoch in range(epochs):
        #     km = self.kernel(self.X_train, self.X_train, **self.kernel_kwargs)
        #     y_pred = self.__predict(X_)

        #     self.v -= alpha * self.gradient(km, y, y_pred)

        #     loss = self.__loss(X_, y)
        #     if loss < self.epsilon: break

        res = minimize(lambda v: self.__emphirical_risk(X_, y, v), x0 = self.v)
        self.v = res.x

    def predict(self, X) -> np.ndarray:
        X_ = self.__get_biased_X(X)
        # return self.__predict_threshold(self.__predict(X_))
        return self.__predict(X_)
    
    def __predict(self, X_) -> np.ndarray:
        km = self.kernel(self.X_train, X_, **self.kernel_kwargs)
        return self.__predict_threshold(self.v @ km)

    def score(self, X, y) -> float:
        X_ = self.__get_biased_X(X)
        return self.__score(X_, y)
    
    def __score(self, X_, y) -> float:
        return (self.__predict(X_) == y).mean()

    def loss(self, X, y) -> float:
        X_ = self.__get_biased_X(X)
        return self.__loss(X_, y)
    
    def __loss(self, X_, y) -> float:
        y_pred = self.__predict(X_)
        return self.logistic_loss(y, y_pred)
    
    def __emphirical_risk(self, X_, y, v) -> float:
        km = self.kernel(self.X_train, X_, **self.kernel_kwargs)
        y_pred = v @ km
        return self.logistic_loss(y, y_pred)

    @staticmethod
    def __predict_threshold(preds, threshold=0) -> np.ndarray:
        return (preds>threshold)*1

    def logistic_loss(self, y, y_pred) -> float:
        return (-y*np.log(self.sigmoid(y_pred)) - (1-y)*np.log(1-self.sigmoid(y_pred))).mean()

    @staticmethod
    def gradient(km, y, y_pred) -> np.ndarray:
        return km @ (y_pred - y)
    
    @staticmethod
    def sigmoid(z) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def __np_array_is_empty_or_null(arr) -> bool:
        return arr is None or not len(arr)
    
    @staticmethod
    def __get_biased_X(X) -> np.ndarray:
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    