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
	if(reg):		# Regression
		models.append(('LiR', LinearRegression(n_jobs=-1, normalize=True)))
		models.append(('LASSO', Lasso(normalize=False, alpha=0.001, selection='cyclic', tol=0.1)))
		models.append(('EN', ElasticNet(normalize=False, alpha=0.001, selection='random', tol=0.2)))
		models.append(('KNN', KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')))
		models.append(('CART', DecisionTreeRegressor(max_features=None, splitter='best', criterion='mse', max_depth=10)))
		models.append(('SVR', SVR()))
		if(ens):
			ensembles.append(('AB', AdaBoostRegressor(DecisionTreeRegressor(max_features=None, splitter='best', criterion='mse', max_depth=10), n_estimators=1000, learning_rate=0.1, loss='linear')))
			ensembles.append(('GBM', GradientBoostingRegressor(loss='ls', learning_rate=1.0, n_estimators=500, max_features='log2')))
			ensembles.append(('RF', RandomForestRegressor(n_jobs=-1, criterion='mse', max_features='log2', n_estimators=20, max_depth=100)))
			ensembles.append(('ET', ExtraTreesRegressor(n_jobs=-1, criterion='mse', max_features=None, n_estimators=20, max_depth=50)))
	elif(clas):		# Classification
		models.append(('LoR', LogisticRegression(multi_class='multinomial', C=0.9, solver='sag')))
		models.append(('MLP', MLPClassifier(alpha=0.001, learning_rate='invscaling', solver='lbfgs')))
		models.append(('LDA', LinearDiscriminantAnalysis(solver='eigen')))
		models.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')))
		models.append(('CART', DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=50)))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC(kernel='rbf', C=14.0)))
		if(ens):
			ensembles.append(('AB', AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=1000, learning_rate=0.1, algorithm='SAMME')))
			ensembles.append(('GBM', GradientBoostingClassifier(max_features='sqrt', loss='deviance', learning_rate=0.1, n_estimators=100)))
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


# Load Data
dataset = read_csv("controller/Tx_full.csv")
#dataset = read_csv("movil/Tx_0x07.csv")
#dataset = read_csv("controller/Tx_mean.csv")
#dataset = read_csv("raspberry/Tx_0x07.csv")
#dataset = read_csv("Tx_full.csv")
# Describe Data
#describeData(dataset)
# Visualize Data
#visualizeData(dataset)

# metrics
seed = 7
num_folds = 10
#scoring = 'accuracy'	# classification
scoring = 'r2'	# regression

# Prepare Data
X_train, X_validation, Y_train, Y_validation = prepareData(dataset, seed)
#Y_train = array([i / 15.0 for i in Y_train])
# Selection of Models
#models = modelSelection(clas=True)
"""
models = []
#models.append(('LoR', LogisticRegression(multi_class='multinomial', C=0.9, solver='sag')))
models.append(('MLP', MLPClassifier(alpha=0.001, learning_rate='invscaling', solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis(solver='eigen')))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')))
models.append(('CART', DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=50)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel='rbf', C=14.0)))
models.append(('AB', AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=1000, learning_rate=0.1, algorithm='SAMME')))
models.append(('GBM', GradientBoostingClassifier(max_features='sqrt', loss='deviance', learning_rate=0.1, n_estimators=100)))
models.append(('RF', RandomForestClassifier(max_features='log2', n_estimators=20, criterion='gini', max_depth=50, class_weight=None)))
models.append(('ET', ExtraTreesClassifier(max_features='sqrt', n_estimators=20, criterion='gini', max_depth=100, class_weight=None)))
# Evaluation of Models
results, names, msg = evaluateAlgorithm(models, num_folds, seed, scoring, X_train, Y_train)
# Show Models Accuracy
print "\n" + (msg)
# Compare Accuraccy of Models
compareAlgorithms(results, names)
"""

"""
# Evaluation of Standarized Models
results, names, msg = evaluateAlgorithm(models, num_folds, seed, scoring, X_train, Y_train)
# Show Standarized Models Accuracy
print(msg)

# Evaluation of Standarized Models
results, names, msg = evaluateAlgorithm(standarizeData(models), num_folds, seed, scoring, X_train, Y_train)
# Show Standarized Models Accuracy
print(msg)
# Compare Accuraccy of Standarized Models
compareAlgorithms(results, names)

# Evaluation of Normalized Models
#results, names, msg = evaluateAlgorithm(normalizeData(models), num_folds, seed, scoring, X_train, Y_train)
# Show Normalized Models Accuracy
#print(msg)
# Compare Accuraccy of Normalized Models
#compareAlgorithms(results, names)

# Tune scaled KNN
# Tune scaled KNN Class
n_neighbors = [1, 3, 5, 7, 9]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
param_grid = dict(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
model = KNeighborsClassifier(n_jobs=-1)
bestKNN = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestKNN)

# Tune scaled SVM
c_values = [13.7, 13.8, 13.9, 14.0, 14.1, 14.2, 14.3]
kernel_values = ['rbf']
param_grid = dict(C=c_values, kernel=kernel_values)
model = SVC()
bestSVM = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestSVM)

# Tune scaled MLP
alpha_values = logspace(-5, 3, 5)
solver_values = ['lbfgs', 'sgd', 'adam']
learning_values = ['adaptive', 'invscaling', 'constant']
param_grid = dict(alpha=alpha_values, solver=solver_values, learning_rate=learning_values)
model = MLPClassifier()
bestMLP = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestMLP)


# Tune scaled LDA
solver = ['eigen', 'lsqr', 'svd']		# ['svd', 'lsqr', 'eigen']
param_grid = dict(solver=solver)
model = LinearDiscriminantAnalysis()
bestLDA = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLDA)

# Tune scaled Log Reg
solver = ['newton-cg', 'lbfgs', 'sag']
multiclass = ['ovr', 'multinomial']
c = [0.8, 0.9, 1.0, 1.1, 1.2]
param_grid = dict(solver=solver, multi_class=multiclass, C=c)
model = LogisticRegression()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

# Tune scaled CART
criterion = ['entropy', 'gini']
max_features = [None, 'sqrt', 'log2']
class_weight = [None, 'balanced']
max_depth = [10, 50, 100]
splitter = ["random", "best"]
param_grid = dict(criterion=criterion, max_features=max_features, class_weight=class_weight, splitter=splitter, max_depth=max_depth)
model = DecisionTreeClassifier()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

# Tune scaled GBClassifier
loss = ['deviance']
learning_rate = [0.01, 0.1, 1.0]
estimators = [100, 300]
max_features = ["sqrt", "log2", None]
param_grid = dict(loss=loss, learning_rate=learning_rate, n_estimators=estimators, max_features=max_features)
model = GradientBoostingClassifier()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)


# Tune scaled AB
algorithm = ['SAMME', 'SAMME.R']
learning_rate = [0.01, 0.1, 1.0, 1.5]
estimators = [600, 800, 1000]
param_grid = dict(algorithm=algorithm, learning_rate=learning_rate, n_estimators=estimators)
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3))
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)


# Tune scaled RF
criterion = ['entropy', 'gini']
max_features = [None, 'sqrt', 'log2']
n_estimators = [5, 10, 20]
class_weight = ['balanced_subsample', None, 'balanced']
max_depth = [10, 50, 100]
param_grid = dict(criterion=criterion, max_features=max_features, n_estimators=n_estimators, class_weight=class_weight, max_depth=max_depth)
model = RandomForestClassifier(n_jobs=-1)
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)


# Tune scaled ET
criterion = ['entropy', 'gini']
max_features = [None, 'sqrt', 'log2']
n_estimators = [5, 10, 20]
class_weight = ['balanced_subsample', None, 'balanced']
max_depth = [10, 50, 100]
param_grid = dict(criterion=criterion, max_features=max_features, n_estimators=n_estimators, class_weight=class_weight, max_depth=max_depth)
model = ExtraTreesClassifier(n_jobs=-1)
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)
"""
"""
#scoring = 'r2'
'LiR'	LinearRegression()
normalize = [True, False]
fit_intercept = [True, False]
copy_X=[True, False]
param_grid = dict(normalize=normalize, fit_intercept=fit_intercept, copy_X=copy_X)
model = LinearRegression()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

'LASSO'	Lasso()
normalize = [True, False]
alpha = [0.001, 0.01, 0.1]
selection = ['cyclic', 'random']
tol = [0.1, 0.3, 0.6]
param_grid = dict(normalize=normalize, alpha=alpha, selection=selection, tol=tol)
model = Lasso()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

'EN'	ElasticNet()
normalize = [True, False]
alpha = [0.001, 0.01, 0.1]
selection = ['cyclic', 'random']
tol = [0.01, 0.1, 0.2]
param_grid = dict(normalize=normalize, alpha=alpha, selection=selection, tol=tol)
model = ElasticNet()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

'KNN'	KNeighborsRegressor()
# Tune scaled KNN Reg
n_neighbors = [1, 3, 5, 7, 9]
weights = ['uniform', 'distance']
algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
param_grid = dict(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
model = KNeighborsRegressor(n_jobs=-1)
bestKNN = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestKNN)

'CART'	DecisionTreeRegressor()
# Tune scaled CART Reg
criterion = ['mse'] #'mae', 
max_features = [None, 'sqrt', 'log2']
max_depth = [5, 10, 15, 20]
splitter = ["random", "best"]
param_grid = dict(criterion=criterion, max_features=max_features, splitter=splitter, max_depth=max_depth)
model = DecisionTreeRegressor()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

'SVR'	SVR()
C = [0.1, 1, 10] #'mae', 
epsilon = [0.1, 0.5, 1.0]
max_depth = [5, 10, 15, 20]
kernel = ["rbf", "linear", "poly", "sigmoid"]
degree = [2, 3, 4, 5]
param_grid = dict(C=C, epsilon=epsilon, kernel=kernel, degree=degree)
model = SVR()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

'AB'	AdaBoostRegressor()
n_estimators=[500,1000]
learning_rate=[0.1, 0.5, 1]
loss=['linear', 'square', 'exponential']
param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss)
model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3))
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)

'GBM'	GradientBoostingRegressor()
'RF'	RandomForestRegressor()
'ET'	ExtraTreesRegressor()

"""
# Tune scaled CART Reg


n_estimators = [500, 1000]
learning_rate = [0.1, 0.5, 1]
loss = ['ls', 'lad', 'huber', 'quantile']
max_depth = [3, 5]
max_features = ["auto", "sqrt", "log2", None]
param_grid = dict(n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, max_depth=max_depth, max_features=max_features)
model = GradientBoostingRegressor()
bestLR = tuneScaledModel(model, X_train, Y_train, num_folds, seed, param_grid, scoring)
tuneResults(bestLR)


#xbee


def makePredictions(modelo, X_train, Y_train, X_validation, Y_validation):
	results = []
	# diffResults = []
	# names = []
	for name, model in modelo:
		# Make predictions on validation dataset
		scaler = StandardScaler().fit(X_train)
		rescaledX = scaler.transform(X_train)
		#model = DecisionTreeClassifier()	 # random_state=seed, n_estimators=400)
		model.fit(rescaledX, Y_train)
		# transform the validation dataset
		rescaledValidationX = scaler.transform(X_validation)
		predictions = model.predict(rescaledValidationX)
		ancho = 3
		distancia = 1.5
		varDist = []
		# [Y_validation, predictions]
		vv = max(predictions) - min(predictions)
		ss = 0
		dif = 0
		s2 = 0
		for i in range(len(predictions)):
			y = Y_validation[i]
			p = round(predictions[i])# * 14 / vv)
			if(y == round(predictions[i])):
				ss = ss + 1
			if(y == round(predictions[i])*14/vv):
				s2 = s2 + 1
			if(y>=predictions[i]-sqrt(2*0.5**2) and y<=predictions[i]+sqrt(2*0.5**2)):
				dif = dif + 1
			vx = fabs(y % ancho - p % ancho)
			vy = fabs(y / ancho - p / ancho)
			varDist.append(sqrt(vx * vx + vy * vy) * distancia)
		err = sum(varDist) / len(varDist)
		mse = mean_squared_error(Y_validation, predictions)
		acc = ss / float(len(predictions))
		print(dif / float(len(predictions)))
		print(s2 / float(len(predictions)))
		print(min(predictions))
		print("\nModelo " + str(name))
		print("MetricErr:\t" + str(err))
		print("MSE:\t\t" + str(mse))
		print("Accuracy:\t" + str(acc))
		# diffResults.append(dif)
		# names.append(name)
		# pyplot.boxplot(dif)
		results.append([name, err, mse, acc])
	# compareAlgorithms(array(diffResults), names)
	# pyplot.boxplot(Y_validation)
	# pyplot.legend(bbox_to_anchor=(1.2, 1))
	# pyplot.show()
	return results

"""

modelos = []
#modelos.append(modelSelection(clas=True))
#modelos.append(modelSelection(clas=True, ens=True))
modelos.append(modelSelection(reg=True))
modelos.append(modelSelection(reg=True, ens=True))

results = []
for modelo in modelos:
	solution = makePredictions(modelo, X_train, Y_train, X_validation, Y_validation)
	for i in range(len(modelo)):
		results.append(solution[i])

columns = ["modelo", "accuracy", "errorMedio"]
DataFrame(array(results)).to_csv("resultados.csv", index=False, header=columns)
"""