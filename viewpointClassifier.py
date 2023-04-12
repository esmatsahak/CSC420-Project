import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    # Load the features and ground truths
    features = np.load('outputs/features.npy')
    gts = np.load('outputs/gts.npy')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, gts, test_size=0.2)

    # Train the model, get test results
    model = MLPClassifier().fit(X_train, y_train)
    model.fit(X_train, y_train)
    print(classification_report(y_test, model.predict(X_test)))