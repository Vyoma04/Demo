from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

MClassifier= MultinomialNB()
MClassifier.fit(X_train, y_train)
MPred = MClassifier.predict(X_test)

BClassifier = BernoulliNB()
BClassifier.fit(X_train, y_train)
BPred = BClassifier.predict(X_test)

print("Accuracies of Naive Bayes' Algorithms with 4 Features:")
print("Multinomial:",accuracy_score(y_test,MPred)*100)
print("Bernoulli:",accuracy_score(y_test,BPred)*100)

A = iris.data[:,:2]
b = iris.target
A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.2)

MClassifier= MultinomialNB()
MClassifier.fit(A_train, b_train)
MPred = MClassifier.predict(A_test)

BClassifier = BernoulliNB()
BClassifier.fit(A_train, b_train)
BPred = BClassifier.predict(A_test)

print("Accuracies of Naive Bayes' Algorithms with 2 Features:")
print("Multinomial:",accuracy_score(y_test,MPred)*100)
print("Bernoulli:",accuracy_score(y_test,BPred)*100)