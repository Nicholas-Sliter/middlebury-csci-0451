import numpy as np

class LogisticRegression:

    def __init__(self, seed = 0, loss_fn = None, grad_fn = None, epsilon = 1e-6):
        self.w = None # weights include bias
        self.__rs = np.random.RandomState(seed)
        self.loss_history = None
        self.score_history = None
        self.epsilon = epsilon

        # Override loss and gradient functions if given by DI
        if loss_fn is not None: self.__loss = loss_fn
        if grad_fn is not None: self.__grad = grad_fn


    def fit(self, X, y, alpha=0.01, epochs=1000):
        if alpha <= 0: raise Exception('Learning rate must be positive')
        if epochs <= 0: raise Exception('Number of epochs must be positive')

        X = self.__get_biased_X(X)
        self.__fit(X, y, alpha, epochs)
    
    def __get_biased_X(self, X) -> np.ndarray:
        ''' Internal method to add bias to X
        @param X: array of shape (n_samples, n_features)
        @return: array of shape (n_samples, n_features + 1)
        '''
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def __get_random_weights(self, n_features) -> np.ndarray:
        ''' Internal method to get random weights
        @param n_features: number of features
        @return: array of shape (n_features,)
        '''
        return self.__rs.rand(n_features)

    def __np_array_is_empty_or_null(self, arr) -> bool:
        ''' Internal method to check if numpy array is empty
        @param arr: numpy array
        @return: bool
        '''
        return (arr is None) or not len(arr)

    def __fit(self, X, y, alpha, epochs) -> None:
        ''' Internal method to fit the model
        @param X: array of shape (n_samples, n_features + 1)
        @param y: array of shape (n_samples,)
        @param alpha: learning rate
        @param epochs: number of epochs
        @return: None
        '''
        if self.w is not None: raise Exception('Model is already trained')

        _, n_features = X.shape
        self.w = self.__get_random_weights(n_features)
        self.__record_history(self.__loss(X, y), self.__score(X, y))

        for epoch in range(epochs):
        
            y_pred = self.__predict(X)

            # Update weights       
            grad = self.__grad(X, y, y_pred)
            update_amt  = (alpha * grad)
            self.w -= update_amt

            # Record loss and check for convergence
            loss = self.__loss(X, y)
            score = self.__score(X, y)

            self.__record_history(loss, score)
            if (np.linalg.norm(update_amt) <= self.epsilon): break
        print(f"Fit model with gradient descent in {epoch+1}/{epochs}(max) epochs.")


    def fit_stochastic(self, X, y, alpha=0.01, epochs=1000, batch_size=100, momentum=False) -> None:
        ''' Fit the model using stochastic gradient descent
        @param X: array of shape (n_samples, n_features)
        @param y: array of shape (n_samples,)
        @param alpha: learning rate
        @param epochs: number of epochs
        @param batch_size: size of batch
        @param momentum: boolean to use momentum
        @return: None
        '''
        if alpha <= 0: raise Exception('Learning rate must be positive')
        if epochs <= 0: raise Exception('Number of epochs must be positive')

        X = self.__get_biased_X(X)
        self.__fit_stochastic(X, y, alpha, epochs, batch_size, momentum)


    def __fit_stochastic(self, X, y, alpha, epochs, batch_size, momentum=False, beta=0.8) -> None:
        ''' Internal method to fit the model using stochastic gradient descent
        @param X: array of shape (n_samples, n_features + 1)
        @param y: array of shape (n_samples,)
        @param alpha: learning rate
        @param epochs: number of epochs
        @param batch_size: size of batch
        @param momentum: boolean to use momentum
        @return: None
        '''
        beta: float = 1*(momentum*beta) # 0 when momentum is false

        if self.w is not None: raise Exception('Model is already trained')

        n, n_features = X.shape
        self.w = self.__get_random_weights(n_features)
        self.__record_history(self.__loss(X, y), self.__score(X, y))

        for epoch in np.arange(epochs):
            order = np.arange(n)
            self.__rs.shuffle(order)

            previous_update_vector = np.zeros(n_features)
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X[batch,:]
                y_batch = y[batch]
                y_batch_pred = self.__predict(x_batch)
                grad = self.__grad(x_batch, y_batch, y_batch_pred) 

                # Stochastic Gradient step with optional momentum
                momentum_term =  (beta * (previous_update_vector))
                update_vector = (alpha * grad) + momentum_term

                previous_update_vector = update_vector

                self.w -= update_vector

            # Record loss and check for convergence
            loss = self.__loss(X, y)
            score = self.__score(X, y)

            self.__record_history(loss, score)
            if (np.linalg.norm(previous_update_vector) <= self.epsilon): break
        print(f"Fit model with stochastic gradient descent in {epoch+1}/{epochs}(max) epochs.")


    
    def __grad(self, X, y, y_pred) -> np.ndarray:
        ''' Internal method to calculate gradient of empirical risk of logistic loss
        @param X: array of shape (n_samples, n_features + 1)
        @param y: array of shape (n_samples,)
        @param y_pred: array of shape (n_samples,)
        @return: array of shape (n_features + 1,)
        '''

        return (1 / len(y)) * ((self.__sigmoid(y_pred) - y) @ X)

    def __record_history(self, loss, score) -> None:
        ''' Internal method to record loss and score history
        @param loss: current loss
        @return: None
        '''
        if self.loss_history is None: self.loss_history = []
        self.loss_history.append(loss)

        if self.score_history is None: self.score_history = []
        self.score_history.append(score)


    def __sigmoid(self, z) -> np.ndarray:
        ''' Internal method to calculate sigmoid
        @param z: array of shape (n_samples,)
        @return: array of shape (n_samples,)
        '''

        return 1 / (1 + np.exp(-z))


    def __loss(self, X, y) -> float:
        ''' Internal method to calculate logistic loss
        @param y: array of shape (n_samples,)
        @param y_pred: array of shape (n_samples,)
        @return: float
        '''
        y_pred = self.__predict(X)

        if self.__np_array_is_empty_or_null(y) or self.__np_array_is_empty_or_null(y_pred): raise Exception('y and y_pred must not be empty')
        if len(y) != len(y_pred): raise Exception('y and y_pred must have same length')

        return (-y*np.log(self.__sigmoid(y_pred)) - (1-y)*np.log(1-self.__sigmoid(y_pred))).mean()


    
    def loss(self, X, y) -> float:
        ''' Logistic loss function
        @param X: array of shape (n_samples, n_features)
        @param y: array of shape (n_samples,)
        @return: float
        '''
        if self.w is None: raise Exception('Model is not trained')
        if not len(X) or not len(y): raise Exception('X and y must not be empty')

        X = self.__get_biased_X(X)
        return self.__loss(X, y)

    def __predict(self, X) -> np.ndarray:
        ''' Internal biased prediction method
        @param X: array of shape (n_samples, n_features + 1)
        @return: array of shape (n_samples,)
        '''
        if self.__np_array_is_empty_or_null(X): raise Exception('X must not be empty')

        predictions = X @ self.w
        return predictions


    def predict(self, X) -> np.ndarray:
        ''' Prediction method
        @param X: array of shape (n_samples, n_features)
        @return: array of shape (n_samples,)
        '''
        if self.w is None: raise Exception('Model is not trained')
        if self.__np_array_is_empty_or_null(X): raise Exception('X must not be empty')

        X = self.__get_biased_X(X)
        return np.array([1 if pred > 0.5 else 0 for pred in self.__predict(X)])

    
    def score(self, X, y) -> float:
        ''' Score method
        @param X: array of shape (n_samples, n_features)
        @param y: array of shape (n_samples,)
        @return: float
        '''
        if self.w is None: raise Exception('Model is not trained')
        if not ((len(X)) and (len(y))): raise Exception('X and y must not be empty')

        y_pred = self.predict(X)
        return (y_pred == y).mean()
    

    def __score(self, X, y) -> float:
        ''' Internal biased score method
        @param X: array of shape (n_samples, n_features + 1)
        @param y: array of shape (n_samples,)
        @return: float
        '''
        if self.__np_array_is_empty_or_null(X) or self.__np_array_is_empty_or_null(y): raise Exception('X and y must not be empty')

        y_pred = np.array([1 if pred > 0.5 else 0 for pred in self.__predict(X)])
        return (y_pred == y).mean()