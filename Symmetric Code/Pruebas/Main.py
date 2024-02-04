# Load libraries
from matplotlib import pyplot
from numpy import arange
from numpy import array
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
from os import system


def menu():
	system('clear')
	print ("Selecciona una opcion")
	print ("\t1 - Algorithm Selection")
	print ("\t2 - segunda opcion")
	print ("\t3 - tercera opcion")
	print ("\t9 - salir")


# Summarize Data
# Descriptive statistics
def describeData(dataset, group_by):
	# shape
	print "Size of data: " + str(dataset.shape)
	# head
	print "\n" + str(dataset.head(10))
	# Types
	print "\n" + str(dataset.dtypes)
	# descriptions
	print "\n" + str(dataset.describe())
	# class distribution
	print "\n" + str(dataset.groupby(group_by).size())
	# correlation
	set_option('precision', 2)
	print "\n" + str(dataset.corr(method='pearson'))


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


# Evaluate Model
def evaluateAlgorithm(models, num_folds, seed, scoring, X_train, Y_train):
	# evaluate each model in turn
	msg = "\n%s\t%s\t%s\n" % ("Algoritmh", scoring, "Desviacion")
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
	Normalize 		data not normally distributed
	Standarize  	data normally distributed
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


# Algorithm Selection
def modelSelection(reg=False, clas=False, ens=False):
	models = []
	ensembles = []
	if(reg):		# Regression
		models.append(('LR', LinearRegression()))
		models.append(('LASSO', Lasso()))
		models.append(('EN', ElasticNet()))
		models.append(('KNN', KNeighborsRegressor()))
		models.append(('CART', DecisionTreeRegressor()))
		models.append(('SVR', SVR()))
		if(ens):
			ensembles.append(('AB', AdaBoostRegressor()))
			ensembles.append(('GBM', GradientBoostingRegressor()))
			ensembles.append(('RF', RandomForestRegressor()))
			ensembles.append(('ET', ExtraTreesRegressor()))
	elif(clas):		# Classification
		models.append(('LR', LogisticRegression()))
		models.append(('MLP', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(15,))))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))
		if(ens):
			ensembles.append(('AB', AdaBoostClassifier()))
			ensembles.append(('GBM', GradientBoostingClassifier()))
			ensembles.append(('RF', RandomForestClassifier()))
			ensembles.append(('ET', ExtraTreesClassifier()))
	if(ens):
		return ensembles
	else:
		return models


"""
	Classification
"""

# Load dataset
#filename = 'iris.data.csv'
#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
filename = 'pima-indians-diabetes.data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = read_csv(filename, names=names)
# filename = 'sonar.all-data.csv'
# dataset = read_csv(filename, header=None)

# metrics
seed = 7
num_folds = 10
scoring = 'accuracy'

# Describe Data
describeData(dataset, group_by=names[-1])

# Visualize Data
# visualizeData(dataset)

# Prepare Data
X_train, X_validation, Y_train, Y_validation = prepareData(dataset, seed)

# Selection of Models
models = modelSelection(clas=True)

# Evaluation of Models
results, names, msg = evaluateAlgorithm(models, num_folds, seed, scoring, X_train, Y_train)
# Show Models Accuracy
print "\n" + (msg)
# Compare Accuraccy of Models
#compareAlgorithms(results, names)

# Evaluation of Standarized Models
results, names, msg = evaluateAlgorithm(standarizeData(models), num_folds, seed, scoring, X_train, Y_train)
# Show Standarized Models Accuracy
print(msg)
# Compare Accuraccy of Standarized Models
#compareAlgorithms(results, names)

# Evaluation of Normalized Models
results, names, msg = evaluateAlgorithm(normalizeData(models), num_folds, seed, scoring, X_train, Y_train)
# Show Normalized Models Accuracy
print(msg)
# Compare Accuraccy of Normalized Models
#compareAlgorithms(results, names)

# Evaluation of Normalized and Standarized Models
results, names, msg = evaluateAlgorithm(standarizeData(normalizeData(models)), num_folds, seed, scoring, X_train, Y_train)
# Show Normalized and Standarized Models Accuracy
print(msg)
# Compare Accuraccy of Normalized and Standarized Models
#compareAlgorithms(results, names)

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

# Ensemble Models
ensembles = modelSelection(clas=True, ens=True)
# Evaluation of Ensemble Models
results, names, msg = evaluateAlgorithm(ensembles, num_folds, seed, scoring, X_train, Y_train)
# Show Ensembled Models Accuracy
print(msg)
# Compare Ensembled Models
compareAlgorithms(results, names)

# Make predictions on validation dataset
print("Test Validacion")
knn = DecisionTreeClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(knn.feature_importances_)
"""
"""
#	Regression
"""
# Load dataset
filename = 'housing.csv'
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = read_csv(filename, delim_whitespace=True, names=names)

# Metrics options
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'		# scoring = 'accuracy'

# Describe Data
describeData(dataset, group_by=names[-1])

# Visualize Data
#visualizeData(dataset)

# Prepare Data
X_train, X_validation, Y_train, Y_validation = prepareData(dataset, seed)

# Selection of Models
models = modelSelection(reg=True)

# Evaluation of Models
results, names, msg = evaluateAlgorithm(models, num_folds, seed, scoring, X_train, Y_train)
# Show Models Accuracy
print "\n" + (msg)
# Compare Accuraccy of Models
#compareAlgorithms(results, names)

# Evaluation of Standarized Models
results, names, msg = evaluateAlgorithm(standarizeData(models), num_folds, seed, scoring, X_train, Y_train)
# Show Standarized Models Accuracy
print(msg)
# Compare Accuraccy of Standarized Models
#compareAlgorithms(results, names)

# Evaluation of Normalized Models
results, names, msg = evaluateAlgorithm(normalizeData(models), num_folds, seed, scoring, X_train, Y_train)
# Show Normalized Models Accuracy
print(msg)
# Compare Accuraccy of Normalized Models
#compareAlgorithms(results, names)

# Evaluation of Normalized and Standarized Models
results, names, msg = evaluateAlgorithm(standarizeData(normalizeData(models)), num_folds, seed, scoring, X_train, Y_train)
# Show Normalized and Standarized Models Accuracy
print(msg)
# Compare Accuraccy of Normalized and Standarized Models
#compareAlgorithms(results, names)

# KNN Algorithm tuning
k_values = array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
bestKNN = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestKNN)

# Tune scaled GBM
param_grid = dict(n_estimators=array([50, 100, 150, 200, 250, 300, 350, 400]))
model = GradientBoostingRegressor(random_state=seed)
bestGBM = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestGBM)

# Ensemble Models
ensembles = modelSelection(reg=True, ens=True)
# Evaluation of Ensemble Models
results, names, msg = evaluateAlgorithm(ensembles, num_folds, seed, scoring, X_train, Y_train)
# Show Ensembled Models Accuracy
print(msg)
# Compare Ensembled Models
compareAlgorithms(results, names)


# Make predictions on validation dataset
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=seed, n_estimators=400)
model.fit(rescaledX, Y_train)
# transform the validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(mean_squared_error(Y_validation, predictions))


"""
print ("Seminario de Tesis I: Algoritmos")
print ("********************************")
opcionMenu = "0"
while opcionMenu != "9":
	menu()
	opcionMenu = raw_input("inserta un numero valor >> ")

	if opcionMenu == "1":
		choose = raw_input("[1]Classification | [2] Regression >> ")

		input("Has pulsado la opcion 1...\npulsa una tecla para continuar")

	elif opcionMenu == "2":
		input("Has pulsado la opcion 2...\npulsa una tecla para continuar")

	elif opcionMenu == "3":
		input("Has pulsado la opcion 3...\npulsa una tecla para continuar")

	else:
		input("No has pulsado ninguna opcion correcta...\npulsa una tecla para continuar")
"""