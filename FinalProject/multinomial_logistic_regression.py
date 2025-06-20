import numpy as np

# Helper function for softmax
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Multinomial Logistic Regression class
class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Ensure X and y are NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # One-hot encode labels
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        # Initialize parameters
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        for _ in range(self.iterations):
            # Compute predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = softmax(linear_model)

            # Gradient descent
            dw = np.dot(X.T, (y_pred - y_one_hot)) / n_samples
            db = np.sum(y_pred - y_one_hot, axis=0, keepdims=True) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = softmax(linear_model)
        return np.argmax(y_pred, axis=1)