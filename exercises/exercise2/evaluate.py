import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from knn import KNN  # Import the KNN class

# Load the dataset
data = pd.read_csv('diabetes.csv').values

X = data[:, :-1]
y = data[:, -1]

X[np.isnan(X)] = np.mean(X, axis=0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def evaluate_knn(k_values):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for k in k_values:
        model = KNN(num_neighbors=k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))
        precisions.append(precision_score(y_test, predictions, zero_division=0))
        recalls.append(recall_score(y_test, predictions, zero_division=0))
        f1_scores.append(f1_score(y_test, predictions))

    return accuracies, precisions, recalls, f1_scores

k_values = range(1, 16)
accuracies, precisions, recalls, f1_scores = evaluate_knn(k_values)

# Plotting the performance metrics
plt.figure(figsize=(10, 8))
plt.plot(k_values, accuracies, label='Accuracy')
plt.plot(k_values, precisions, label='Precision')
plt.plot(k_values, recalls, label='Recall')
plt.plot(k_values, f1_scores, label='F1 Score')
plt.xlabel('Number of Neighbors k')
plt.ylabel('Performance')
plt.legend()
plt.title('k-NN Performance Evaluation with Different k')
plt.show()
