def euclidean_distance(row1, row2):
    """
    Calculate the Euclidean distance between two data points.
    """
    distance = np.sqrt(np.sum((row1 - row2) ** 2))
    return distance

def predict_classification(train, test_row, num_neighbors):
    """
    Predict the class for a test row based on majority voting from k-nearest neighbors.
    """
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row[:-1])
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = distances[:num_neighbors]

    output_values = [row[0][-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

