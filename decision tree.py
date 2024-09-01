import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the decision tree classifier
    clf = tree.DecisionTreeClassifier(criterion='gini', random_state=42)
    clf = clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Visualize the decision tree
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
    plt.show()

if __name__ == "__main__":
    main()
