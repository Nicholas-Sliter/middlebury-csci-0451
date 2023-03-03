import numpy as np
import numpy.typing as npt

class Perceptron:
    def __init__(self):
        self.w = None
        self.history = None

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], max_steps = 10000, rate = 1, epsilon: float = 0) -> None:
        ''' Fit the model to the given data
        @param X: 2D array of shape (n_samples, n_features) of data
        @param y: 1D array of shape (n_samples) of labels (0 or 1)
        @param max_steps: int of the maximum number of steps to run the algorithm (default 10000)
        @param rate: float of the learning rate (default 1)
        @param epsilon: float of the loss threshold to stop at (default 0)
        @return: None
        '''
        self.w = np.random.rand(X.shape[1] + 1) # add bias
        self.history = []

        X_ = self.__get_biased_X(X)

        i = 0
        while (not self.__has_epsilon_converged(X_, y, epsilon)) and i < max_steps:
            self.__update_weights(X_, y, self.w, rate)
            self.__record_history(X_, y)
            i += 1
        
        if (i == 0): # model can be fit by random guess
            self.__record_history(X_, y)

        print(f"Fit model in {i} steps with score {self.history[-1]}")

    def predict(self, X) -> np.ndarray:
        ''' Predict the class of each sample in X
        @param X: 2D array of shape (n_samples, n_features)
        @return: 1D array of shape (n_samples) of predictions (0 or 1)
        '''
        X_ = self.__get_biased_X(X)
        return self.__predict(X_)

    def score(self, X, y) -> float:
        ''' Score the model on the given data
        @param X: 2D array of shape (n_samples, n_features) of data
        @param y: 1D array of shape (n_samples) of labels (0 or 1)
        @return: float of the accuracy score (0 to 1)
        '''
        predictions = self.predict(X)
        return np.sum(predictions == y) / len(y)

    def __predict(self, X) -> np.ndarray:
        ''' Internal predict method that assumes X has bias
        @param X: 2D array of shape (n_samples, n_features + 1)
        @return: 1D array of shape (n_samples) of predictions (0 or 1)
        '''
        if self.w is None:
            raise Exception("Model not trained yet")
        if X.shape[1] != self.w.shape[0]:
            raise Exception("X shape does not match model shape")
        
        return 1*((X @ self.w) >= 0) # converts bool to int


    def __get_biased_X(self, X) -> np.ndarray:
        ''' Internal method to add bias to X
        @param X: 2D array of shape (n_samples, n_features)
        @return: 2D array of shape (n_samples, n_features + 1)
        '''
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def __score(self, X, y) -> float:
        ''' Internal score method that assumes X has bias
        @param X: 2D array of shape (n_samples, n_features + 1)
        @param y: 1D array of shape (n_samples) of labels (0 or 1)
        @return: float of the accuracy score (0 to 1)
        '''
        predictions = self.__predict(X)
        return np.sum(predictions == y) / len(y)

    def __has_epsilon_converged(self, X, y, epsilon) -> bool:
        ''' Internal method to check for convergence
        @param X: 2D array of shape (n_samples, n_features + 1)
        @param y: 1D array of shape (n_samples) of labels (0 or 1)
        @param epsilon: float of the threshold to stop at
        @return: bool of whether the score is <= epsilon
        '''
        loss = 1 - self.__score(X, y)
        return loss <= epsilon

    def __update_weights(self, X, y, w, rate) -> None:
        ''' Internal method to update model weights
        @param X: 2D array of shape (n_samples, n_features + 1)
        @param y: 1D array of shape (n_samples) of labels (0 or 1)
        @param w: 1D array of shape (n_features + 1) of weights
        @param rate: float of the learning rate
        @return: None
        '''
        index = np.random.randint(0, len(X))
        y_hat = 2 * y[index] - 1

        updated_weights = w + int(y_hat * np.dot(X[index], w) < 0) * rate * (X[index] * y_hat)
        self.w = updated_weights
        
    def __record_history(self, X, y) -> None:
        ''' Internal method to record the score of the model
        @param X: 2D array of shape (n_samples, n_features + 1)
        @param y: 1D array of shape (n_samples) of labels (0 or 1)
        @return: None
        '''
        if self.history is None:
            self.history = []
        self.history.append(self.__score(X, y))
