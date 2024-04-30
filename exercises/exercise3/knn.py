import numpy as np

class KNN:
    def __init__(self, num_neighbors=3):
        self.num_neighbors = num_neighbors

    def euclidean_distance(self, row1, row2):
        """
        Calculate the Euclidean distance between two data points.
        """
        distance = np.sqrt(np.sum((row1 - row2) ** 2))
        return distance

    def fit(self, X, y):
        """
        Fit the training data to the model.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class for each data point in X.
        """
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        """
        Predict the class for a single data point x.
        """
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.num_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common
