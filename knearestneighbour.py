import random 
from scipy.spatial import distance
from scipy.stats import mode
import numpy


def euc(a, b):
    return distance.euclidean(a, b)


class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        dist = []
        
        for i in range(1,len(self.X_train)):
            dist.append([euc(row, self.X_train[i]), i])
        dist.sort(key=lambda dist: dist[0])
        k_smallest = dist[:3]
        k_nearest_labels = [self.y_train[i] for distance, i in k_smallest]
        best_index = mode(k_nearest_labels)
        
        return [int(best_index.mode)]
            

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)



# classifying model 
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
