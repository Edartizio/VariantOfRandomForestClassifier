import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import pairwise_distances
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class VariantOfRandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.classifiers = []

    def fit(self, X, y):
        self.classifiers = []
        
        # Iterazione per creare e addestrare gli alberi decisionali
        for _ in range(self.n_estimators):
            dataset_indices = self._create_dataset(X)
            dataset_X, dataset_y = X[dataset_indices], y[dataset_indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(dataset_X, dataset_y)

            self.classifiers.append(tree)

    def predict_proba(self, X):
        # Calcolo delle probabilità medie
        all_proba = np.zeros((X.shape[0], len(self.classifiers)))

        for i, tree in enumerate(self.classifiers):
            tree_proba = tree.predict_proba(X)
            all_proba[:, i] = tree_proba[:, 1]  

        return np.average(all_proba, axis=1)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)  

    def _create_dataset(self, X):
        seed_index = np.random.choice(len(X))
        seed_instance = X[seed_index]

        distances = pairwise_distances(X, [seed_instance])
        similarities = 1 / (1 + distances)

        weights = similarities / np.sum(similarities)

        selected_indices = np.random.choice(len(X), size=len(X), replace=True, p=weights.flatten())

        return selected_indices


iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


hf = VariantOfRandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
hf.fit(X_train, y_train)

predictions = hf.predict(X_test)
probabilities = hf.predict_proba(X_test)
print("Predizioni = ", predictions)
print("Probabilità = ", probabilities)

accuracy = accuracy_score(y_test, predictions)
print("Accuratezza", accuracy)