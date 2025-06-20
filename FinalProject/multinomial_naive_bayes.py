import numpy as np

# Multinomial Naive Bayes class
class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.feature_log_probs = None

    def fit(self, X, y):
        # Ensure X and y are NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        self.class_priors = np.zeros(n_classes)
        self.feature_log_probs = np.zeros((n_classes, n_features))

        for c in range(n_classes):
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / n_samples
            self.feature_log_probs[c, :] = np.log((X_c.sum(axis=0) + 1) / (X_c.sum() + n_features))

    def predict(self, X):
        # Ensure X is a NumPy array
        X = np.asarray(X)
        log_probs = np.dot(X, self.feature_log_probs.T) + np.log(self.class_priors)
        return np.argmax(log_probs, axis=1)