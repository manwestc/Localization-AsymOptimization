import numpy as np
import pylab as pl

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Linear Classification
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
# NonLinear Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# Ensemble Classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
# NonLinear Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
# Ensemble Regression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from math import sqrt
from math import ceil
from os import system


def plot_classification_results(clf, X, y, title):
	# Divide dataset into training and testing parts
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
	# Fit the data with classifier.
	clf.fit(X_train, y_train)
	# Create color maps
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
	h = .02  # step size in the mesh
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, m_max]x[y_min, y_max].
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	pl.figure()
	pl.pcolormesh(xx, yy, Z, cmap=cmap_light)
	# Plot also the training points
	pl.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold)
	y_predicted = clf.predict(X_test)
	score = clf.score(X_test, y_test)
	pl.scatter(X_test[:, 0], X_test[:, 1], c=y_predicted, alpha=0.5, cmap=cmap_bold)
	pl.xlim(xx.min(), xx.max())
	pl.ylim(yy.min(), yy.max())
	pl.title(title)
	return score


xs = load_iris().data[:, :2]
ys = load_iris().target
models = [("LDA", LinearDiscriminantAnalysis()),
		("Logistic Regression", LogisticRegression()),
		("Neural Network", MLPClassifier())]

for name, clf in models:
	score = plot_classification_results(clf, xs, ys, name + " classification")
	print ("%s score: %s" % (name, score))
pl.show()
