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
        self.dataset_seed= None

    def fit(self, X, y):
        self.classifiers = []
        
        # Iterazione per creare e addestrare gli alberi decisionali
        for _ in range(self.n_estimators):
            # Seleziona casualmente il seed in base alla dissimilarità media
            seed_index = self.selected_seed(X)
            seed_istances = X[seed_index]
            self.dataset_seed = seed_istances

            dataset_indices = self.create_dataset(X, seed_istances)
            dataset_X, dataset_y = X[dataset_indices], y[dataset_indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(dataset_X, dataset_y)

            self.classifiers.append(tree)

    def predict_proba(self, X):
        all_proba = np.zeros((X.shape[0], 2))  # 2 classi: 0 e 1

        for i, x in enumerate(X):
            # Calcola le distanze rispetto al dataset selezionato in precedenza
            distances = pairwise_distances([x], [self.dataset_seed])
            similarities = 1 / (1 + distances)

            # Utilizza la similarità normalizzata come peso per le predizioni degli alberi
            weights = similarities / np.sum(similarities)

            # Calcola le probabilità predette dai vari alberi
            tree_probas = np.array([[tree.predict_proba([x])[0, 1]] for tree in self.classifiers])

            
            flattened_weights = weights.flatten()

            # Mediare le predizioni degli alberi utilizzando i pesi
            weighted_probas = np.average(tree_probas, weights=flattened_weights)

            
            all_proba[i, 1] = weighted_probas

        # La colonna 0 rappresenta la probabilità della classe 0
        all_proba[:, 0] = 1 - all_proba[:, 1]

        return all_proba


    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def selected_seed(self, X):
        # Calcola le dissimilirità medie rispetto ai dataset selezionati in precedenza 
        if len(self.classifiers) == 0:
            # Se è il primo dataset, seleziona casualmente il seed
            return np.random.choice(len(X))
        else:
            # Calcola le dissimilarità medie rispetto ai dataset selezionati in precedenza
            distances_sum = np.sum([pairwise_distances(X, np.array([self.dataset_seed])) for _ in self.classifiers], axis=0)
            average_distances = distances_sum / len (self.classifiers)

            # Seleziona il seed in base alla dissimilarità media
            seed_index = np.argmax(np.mean(average_distances, axis=1))
            return seed_index
        
    def create_dataset(self, X, seed_istances):

        distances = pairwise_distances(X, [seed_istances])
        similarities = 1 / (1 + distances)

        weights = similarities / np.sum(similarities)

        # Campiona casualmente gli indici del dataset basandosi sui pesi
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