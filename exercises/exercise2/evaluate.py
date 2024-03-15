import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from exercises.exercise2.knn import  euclidean_distance,  predict_classification


# Assuming the dataset is loaded into a NumPy array 'data'
# Features are in columns 0-7, and the label is in column 8

# Splitting dataset into features and target variable
X = data[:, :-1]
y = data[:, -1]

# Handling missing values (if any)
# For simplicity, replacing missing values with the mean of each column
X[np.isnan(X)] = np.mean(X, axis=0)

# Normalizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Function to evaluate k-NN with different values of k
def evaluate_knn(k_values):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    # Combine training features and target for simplicity in k-NN calculation
    training_data = np.column_stack((X_train, y_train))

    for k in k_values:
        predictions = [predict_classification(training_data, row, k) for row in X_test]
        accuracies.append(accuracy_score(y_test, predictions))
        precisions.append(precision_score(y_test, predictions))
        recalls.append(recall_score(y_test, predictions))
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
