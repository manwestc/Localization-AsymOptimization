# Load libraries
from matplotlib import pyplot
from math import sqrt
from math import ceil
from math import fabs
from numpy import arange
from numpy import array
from numpy import isnan
from numpy import where
from numpy import take
from numpy import vstack
from numpy import logspace
from scipy.stats import mode
from scipy.stats import nanmean
from scipy.stats import describe
from pandas import read_csv
from pandas import set_option
from pandas import DataFrame
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
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


# Summarize Data
def describeData(dataset):
	# shape
	print "\nSize of data: " + str(dataset.shape)
	# head
	# print "\n" + str(dataset.head(10))
	# Types
	print "\nDataTypes:\n" + str(dataset.dtypes)
	# descriptions
	print "\nDataDescription:\n" + str(dataset.describe())
	# class distribution
	print "Dataset Group By Position\n" + str(dataset.groupby(dataset.columns.values[-1]).size())
	# correlation
	set_option('precision', 2)
	print "Pearson Correlation Matrix\n" + str(dataset.corr(method='pearson'))


# Data visualizations
def visualizeData(dataset):
	layout = (int(ceil(sqrt(dataset.shape[1]))), int(round(sqrt(dataset.shape[1]))))
	# box and whisker plots
	dataset.plot(kind='box', subplots=True, layout=layout, sharex=False, sharey=False)
	# density
	dataset.plot(kind='density', subplots=True, layout=layout, sharex=False, sharey=False)
	# histograms
	dataset.hist()
	# DataFrame(dataset.values[:,:-1],columns=["BLE1","BLE2","BLE3","BLE4","BLE5"]).plot(kind='box', sharex=True, sharey=True)
	DataFrame
	# scatter plot matrix
	scatter_matrix(dataset)
	# correlation Matrix
	fig = pyplot.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
	fig.colorbar(cax)
	pyplot.show()


# Prepare Data
def prepareData(dataset, seed=7):
	# Split-out validation dataset
	vector = dataset.values
	X = vector[:, 0:dataset.shape[1] - 1].astype(float)
	Y = vector[:, dataset.shape[1] - 1]
	validation_size = 0.20
	return train_test_split(X, Y, test_size=validation_size, random_state=seed)


# Algorithm Selection
def modelSelection(reg=False, clas=False, ens=False):
	models = []
	ensembles = []
	if(reg):                # Regression
		models.append(('LiR', LinearRegression(n_jobs=-1, normalize=True)))
		models.append(('LASSO', Lasso(normalize=False, alpha=0.001, selection='cyclic', tol=0.1)))
		models.append(('EN', ElasticNet(normalize=False, alpha=0.001, selection='random', tol=0.2)))
		models.append(('KNN', KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')))
		models.append(('CART', DecisionTreeRegressor(max_features=None, splitter='best', criterion='mse', max_depth=10)))
		models.append(('SVR', SVR()))
		if(ens):
			ensembles.append(('AB', AdaBoostRegressor()))
			ensembles.append(('GBM', GradientBoostingRegressor()))
			ensembles.append(('RF', RandomForestRegressor()))
			ensembles.append(('ET', ExtraTreesRegressor()))
	elif(clas):             # Classification
		models.append(('LoR', LogisticRegression(multi_class='multinomial', C=0.9, solver='sag')))
		models.append(('MLP', MLPClassifier(alpha=0.001, learning_rate='invscaling', solver='lbfgs')))
		models.append(('LDA', LinearDiscriminantAnalysis(solver='eigen')))
		models.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')))
		models.append(('CART', DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=50)))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC(kernel='rbf', C=14.0)))
		if(ens):
			ensembles.append(('AB', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=1000, learning_rate=0.1, algorithm='SAMME')))
			ensembles.append(('GBM', GradientBoostingClassifier(max_features='sqrt', loss='desviance', learning_rate=0.1, n_estimators=100)))
			ensembles.append(('RF', RandomForestClassifier(max_features='log2', n_estimators=20, criterion='gini', max_depth=50, class_weight=None)))
			ensembles.append(('ET', ExtraTreesClassifier(max_features='sqrt', n_estimators=20, criterion='gini', max_depth=100, class_weight=None)))
	if(ens):
		return ensembles
	else:
		return models


# Evaluate Model
def evaluateAlgorithm(models, num_folds, seed, scoring, X_train, Y_train):
	# evaluate each model in turn
	msg = "\n%s\t%s\t%s\n" % ("model", scoring, "desviation")
	results = []
	names = []
	for name, model in models:
		kfold = KFold(n_splits=num_folds, random_state=seed)
		# estimators.append((name, model))
		# modelPipeline = Pipeline(estimators)
		cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
		# estimators.pop()
		results.append(cv_results)
		names.append(name)
		msg = msg + "%s: \t%f\t(%f)\n" % (name, cv_results.mean(), cv_results.std())
		# print(msg)
	return results, names, msg


# Compare Algorithms
def compareAlgorithms(results, names):
	fig = pyplot.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	pyplot.boxplot(results)
	ax.set_xticklabels(names)
	pyplot.show()


"""
	Scale Data Methods
	Normalize               data not normally distributed
	Standarize      data normally distributed
"""


# Standardize the dataset
def standarizeData(models):
	pipelines = []
	for name, model in models:
		pipelines.append(('Scaled' + name, Pipeline([('Scaler', StandardScaler()), (name, model)])))
	return pipelines


# Normalize the dataset
def normalizeData(models):
	pipelines = []
	for name, model in models:
		pipelines.append(('Norm' + name, Pipeline([('Normalizer', Normalizer()), (name, model)])))
	return pipelines


# Tune scaled Model
def tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring):
	scaler = StandardScaler().fit(X_train)
	rescaledX = scaler.transform(X_train)
	kfold = KFold(n_splits=num_folds, random_state=seed)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
	grid_result = grid.fit(rescaledX, Y_train)
	return grid_result


# Show Tune Results
def tuneResults(grid_result):
	print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))


# Load Data
#dataset = read_csv("controller/Tx_full.csv")
#dataset = read_csv("movil/Tx_0x07.csv")
#dataset = read_csv("controller/Tx_mean.csv")
#dataset = read_csv("raspberry/Tx_0x07.csv")
#dataset = read_csv("Tx_full.csv")
# Describe Data
#describeData(dataset)
# Visualize Data
#visualizeData(dataset)



classifier = []
classifier.append(('LoR', LogisticRegression(multi_class='multinomial', C=0.9, solver='sag')))
classifier.append(('MLP', MLPClassifier(alpha=0.001, learning_rate='invscaling', solver='lbfgs')))
classifier.append(('LDA', LinearDiscriminantAnalysis(solver='eigen')))
classifier.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')))
classifier.append(('CART', DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=50)))
classifier.append(('NB', GaussianNB()))
classifier.append(('SVM', SVC(kernel='rbf', C=14.0)))
classifier.append(('AB', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=1000, learning_rate=0.1, algorithm='SAMME')))
classifier.append(('GBM', GradientBoostingClassifier(max_features='sqrt', loss='desviance', learning_rate=0.1, n_estimators=100)))
classifier.append(('RF', RandomForestClassifier(max_features='log2', n_estimators=20, criterion='gini', max_depth=50, class_weight=None)))
classifier.append(('ET', ExtraTreesClassifier(max_features='sqrt', n_estimators=20, criterion='gini', max_depth=100, class_weight=None)))

seed = 7
num_folds = 10
scoring = 'accuracy'

# Prepare Data
#X_train, X_validation, Y_train, Y_validation = prepareData(dataset, seed)

#results, names, msg = evaluateAlgorithm(standarizeData(classifier), num_folds, seed, scoring, X_train, Y_train)



dataset1 = read_csv("raspberry/Tx_0x01.csv")
dataset2 = read_csv("raspberry/Tx_0x02.csv")
dataset3 = read_csv("raspberry/Tx_0x03.csv")
dataset4 = read_csv("raspberry/Tx_0x04.csv")
dataset5 = read_csv("raspberry/Tx_0x05.csv")
dataset6 = read_csv("raspberry/Tx_0x06.csv")
dataset7 = read_csv("raspberry/Tx_0x07.csv")
layout = (3, 2)
#dataset = read_csv("controller/Tx_full.csv")
#DataFrame(dataset2.values[:,:-1],columns=["BLE1","BLE2","BLE3","BLE4","BLE5"]).plot(kind='box', sharex=True, sharey=True)
#pyplot.show()

datasets = [dataset1,dataset2,dataset3,dataset4,dataset5,dataset6,dataset7]
for dataset in datasets:
	describeData(dataset)
	#DataFrame(dataset.values[:,:-1],columns=["BLE1","BLE2","BLE3","BLE4","BLE5"]).plot(kind='density', sharex=True, sharey=True, subplots=True, xlim=[-120,-49], layout=(2,3))

#pyplot.subplot()
#DataFrame(dataset.values[:,-1],columns=["Sector"]).plot(kind='density', sharex=False, sharey=False)

pyplot.show()

"""
import numpy as np
def menor_distancia(truth, predictions):
	width = 3
	height = 5
	if truth == predictions:
		return 0
	else:
		dx = np.abs(predictions%width-truth%width)*1.5-1
		dy = np.abs(predictions/width-truth/width)*1.5-1
		return np.sqrt(dx**2+dy**2)

loss  = make_scorer(menor_distancia, greater_is_better=False)
"""

"""

% Matplotlib
% http://pendientedemigracion.ucm.es/info/aocg/python/modulos_cientificos/matplotlib/index.html
% grafica funcion logistica
% -------------------------
from scipy import linspace
from  matplotlib import pyplot
from scipy import sin
from scipy import exp
x = linspace(-6,6,100)
y = 1/(1+exp(-x))
pyplot.figure()
pyplot.plot(x,y)
pyplot.show()

"""
"""

Logistic regression is a simple and powerful linear classification algorithm. It also has limitations
that suggest at the need for alternate linear classification algorithms.
* Two-Class Problems. Logistic regression is intended for two-class or binary classification
problems. It can be extended for multiclass classification, but is rarely used for this purpose.
* Unstable With Well Separated Classes. Logistic regression can become unstable
when the classes are well separated.
* Unstable With Few Examples. Logistic regression can become unstable when there
are few examples from which to estimate the parameters.


The advantages of Multi-layer Perceptron are:

	Capability to learn non-linear models.
	Capability to learn models in real-time (on-line learning) using partial_fit.

The disadvantages of Multi-layer Perceptron (MLP) include:

	MLP with hidden layers have a non-convex loss function where there exists more than one local minimum. Therefore different random weight initializations can lead to different validation accuracy.
	MLP requires tuning a number of hyperparameters such as the number of hidden neurons, layers, and iterations.
	MLP is sensitive to feature scaling.



Some advantages of decision trees are:

	Simple to understand and to interpret. Trees can be visualised.
	Requires little data preparation. Other techniques often require data normalisation, dummy variables need to be created and blank values to be removed. Note however that this module does not support missing values.
	The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
	Able to handle both numerical and categorical data. Other techniques are usually specialised in analysing datasets that have only one type of variable. See algorithms for more information.
	Able to handle multi-output problems.
	Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.
	Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.
	Performs well even if its assumptions are somewhat violated by the true model from which the data were generated.

The disadvantages of decision trees include:

	Decision-tree learners can create over-complex trees that do not generalise the data well. This is called overfitting. Mechanisms such as pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
	Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
	The problem of learning an optimal decision tree is known to be NP-complete under several aspects of optimality and even for simple concepts. Consequently, practical decision-tree learning algorithms are based on heuristic algorithms such as the greedy algorithm where locally optimal decisions are made at each node. Such algorithms cannot guarantee to return the globally optimal decision tree. This can be mitigated by training multiple trees in an ensemble learner, where the features and samples are randomly sampled with replacement.
	There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
	Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.

"""