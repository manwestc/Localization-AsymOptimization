# Hello World Classification: Iris flowers prediction

# Prepare Problem

# Load libraries
from matplotlib import pyplot
from numpy import arange
from pandas import read_csv
from pandas import set_option
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

from math import sqrt
from math import ceil

"""
Small dataset or L1 penalty			"liblinear"
Multinomial loss or large dataset 	"lbfgs", "sag" or "newton-cg"
Very Large dataset 					"sag"
"""

# Load dataset
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# filename = 'pima-indians-diabetes.data.csv'
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = read_csv(filename, names=names)
# filename = 'sonar.all-data.csv'
# dataset = read_csv(filename, header=None)


# Summarize Data
# Descriptive statistics
def describeData(dataset, group_by='class'):
	# shape
	print(dataset.shape)
	# head
	print(dataset.head(20))
	# Types
	print(dataset.dtypes)
	# descriptions
	print(dataset.describe())
	# class distribution
	print(dataset.groupby(group_by).size())
	# correlation
	set_option('precision', 2)
	print(dataset.corr(method='pearson'))


# Data visualizations
def visualizeData(dataset):
	layout = (int(ceil(sqrt(dataset.shape[1]))), int(round(sqrt(dataset.shape[1]))))
	# box and whisker plots
	dataset.plot(kind='box', subplots=True, layout=layout, sharex=False, sharey=False)
	# density
	dataset.plot(kind='density', subplots=True, layout=layout, sharex=False, sharey=False)
	# histograms
	dataset.hist()
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
	array = dataset.values
	X = array[:, 0:dataset.shape[1] - 1].astype(float)
	Y = array[:, dataset.shape[1] - 1]
	validation_size = 0.20
	return train_test_split(X, Y, test_size=validation_size, random_state=seed)


seed = 7
describeData(dataset, group_by='class')
# visualizeData(dataset)
X_train, X_validation, Y_train, Y_validation = prepareData(dataset, seed)

# Test options and evaluation metric
num_folds = 10
scoring = 'accuracy'
"""
# create feature union
features = []
features.append(('pca', PCA(n_components=4)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('standardize', StandardScaler()))
# estimators.append(('normalize', Normalizer()))
estimators.append(('scale', MinMaxScaler(feature_range=(0, 1))))
# Spot-Check Algorithms
"""


def evaluateAlgorithm(models, num_folds, seed, scoring, X_train, Y_train):
	# evaluate each model in turn
	msg = "%s\t%s\t%s\n" % ("Algoritmo", "Accuracy", "Desviacion")
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


# Classifier Models
def classificationModels():
	models = []
	models.append(('LR', LogisticRegression()))
	models.append(('MLP', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(15,))))
	models.append(('LDA', LinearDiscriminantAnalysis()))
	models.append(('KNN', KNeighborsClassifier()))
	models.append(('CART', DecisionTreeClassifier()))
	models.append(('NB', GaussianNB()))
	models.append(('SVM', SVC()))


models = classificationModels()

results, names, msg = evaluateAlgorithm(models, num_folds, seed, scoring, X_train, Y_train)

print(msg)


# Compare Algorithms
def compareAlgorithms(results, names):
	fig = pyplot.figure()
	fig.suptitle('Algorithm Comparison')
	ax = fig.add_subplot(111)
	pyplot.boxplot(results)
	ax.set_xticklabels(names)
	pyplot.show()


# Standardize the dataset
def standarizeData(models):
	pipelines = []
	for name, model in models:
		pipelines.append(('Scaled' + name, Pipeline([('Scaler', StandardScaler()), (name, model)])))
	return pipelines


# no standarized
compareAlgorithms(results, names)
results, names, msg = evaluateAlgorithm(standarizeData(models), num_folds, seed, scoring, X_train, Y_train)
print(msg)
# estandarized
compareAlgorithms(results, names)


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
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))


# Tune scaled KNN
neighbors = range(1, 20)  # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
param_grid = dict(n_neighbors=neighbors)
model = KNeighborsClassifier()
bestKNN = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestKNN)

# Tune scaled SVM
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
bestSVM = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestSVM)


# ensembles
def ensembledClassifiers():
	ensembles = []
	ensembles.append(('AB', AdaBoostClassifier()))
	ensembles.append(('GBM', GradientBoostingClassifier()))
	ensembles.append(('RF', RandomForestClassifier()))
	ensembles.append(('ET', ExtraTreesClassifier()))
	return ensembles


ensembles = ensembledClassifiers()

results, names, msg = evaluateAlgorithm(ensembles, num_folds, seed, scoring, X_train, Y_train)
print(msg)
compareAlgorithms(results, names)
"""
# Make predictions on validation dataset
print("Test Validacion")
knn = DecisionTreeClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(knn.feature_importances_)


# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=1.5)
model.fit(rescaledX, Y_train)
# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

"""
