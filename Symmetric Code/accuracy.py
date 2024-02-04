# Load libraries
from matplotlib import pyplot
from itertools import compress
from math import sqrt
from math import ceil
from math import fabs
from numpy import arange
from numpy import array
from numpy import isnan
from numpy import where
from numpy import take
from numpy import ceil
from numpy import floor
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
	#DataFrame
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
	vector = dataset.values
	X = vector[:, 0:dataset.shape[1] - 1].astype(float)
	Y = vector[:, dataset.shape[1] - 1]
	test_size = 0.20
	return train_test_split(X, Y, test_size=test_size, random_state=seed)


# Algorithm Selection
def modelSelection(reg=False, clas=False, ens=False):
	models = []
	if(reg):		# Regression
		models.append(('LiR', LinearRegression(n_jobs=-1, normalize=True)))
		models.append(('LASSO', Lasso(normalize=False, alpha=0.001, selection='cyclic', tol=0.1)))
		models.append(('EN', ElasticNet(normalize=False, alpha=0.001, selection='random', tol=0.2)))
		models.append(('KNN', KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')))
		models.append(('CART', DecisionTreeRegressor(max_features=None, splitter='best', criterion='mse', max_depth=10)))
		models.append(('SVR', SVR(kernel='sigmoid', C=14.0)))
		if(ens):
			models.append(('AB', AdaBoostRegressor(DecisionTreeRegressor(max_features=None, splitter='best', criterion='mse', max_depth=10), n_estimators=1000, learning_rate=0.1, loss='linear')))
			models.append(('GBM', GradientBoostingRegressor(loss='ls', learning_rate=1.0, n_estimators=500, max_features='log2')))
			models.append(('RF', RandomForestRegressor(n_jobs=-1, criterion='mse', max_features='log2', n_estimators=20, max_depth=100)))
			models.append(('ET', ExtraTreesRegressor(n_jobs=-1, criterion='mse', max_features=None, n_estimators=20, max_depth=50)))
	if(clas):		# Classification
		models.append(('LoR', LogisticRegression(multi_class='multinomial', C=0.9, solver='sag')))
		models.append(('MLP', MLPClassifier(alpha=0.001, learning_rate='invscaling', solver='lbfgs')))
		models.append(('LDA', LinearDiscriminantAnalysis(solver='eigen')))
		models.append(('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')))
		models.append(('CART', DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=50)))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC(kernel='rbf', C=14.0)))
		if(ens):
			models.append(('AB', AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=1000, learning_rate=0.1, algorithm='SAMME')))
			models.append(('GBM', GradientBoostingClassifier(max_features='sqrt', loss='deviance', learning_rate=0.1, n_estimators=100)))
			models.append(('RF', RandomForestClassifier(max_features='log2', n_estimators=20, criterion='gini', max_depth=50, class_weight=None)))
			models.append(('ET', ExtraTreesClassifier(max_features='sqrt', n_estimators=20, criterion='gini', max_depth=100, class_weight=None)))
	return models


# Algorithm Selection Clear
def modelClear(reg=False, clas=False, ens=False):
	models = []
	if(reg):		# Regression
		models.append(('LiR', LinearRegression()))
		models.append(('LASSO', Lasso()))
		models.append(('EN', ElasticNet()))
		models.append(('KNN', KNeighborsRegressor()))
		models.append(('CART', DecisionTreeRegressor()))
		models.append(('SVR', SVR(kernel='linear', C=14.0)))
		if(ens):
			models.append(('AB', AdaBoostRegressor()))
			models.append(('GBM', GradientBoostingRegressor()))
			models.append(('RF', RandomForestRegressor()))
			models.append(('ET', ExtraTreesRegressor()))
	if(clas):		# Classification
		models.append(('LoR', LogisticRegression()))
		models.append(('MLP', MLPClassifier()))
		models.append(('LDA', LinearDiscriminantAnalysis()))
		models.append(('KNN', KNeighborsClassifier()))
		models.append(('CART', DecisionTreeClassifier()))
		models.append(('NB', GaussianNB()))
		models.append(('SVM', SVC()))
		if(ens):
			models.append(('AB', AdaBoostClassifier()))
			models.append(('GBM', GradientBoostingClassifier()))
			models.append(('RF', RandomForestClassifier()))
			models.append(('ET', ExtraTreesClassifier()))
	return models


# Algorithm Pipeline
def modelFull(reg=False, clas=False, ens=False):
	models = []
	if(reg):		# Regression
		models.append(('LiR', Pipeline([('nor', Normalizer()), ('std', StandardScaler()), ('LiR', LinearRegression(n_jobs=-1, normalize=True))])))
		models.append(('Lasso', Pipeline([('nor', Normalizer()), ('std', StandardScaler()), ('LASSO', Lasso(normalize=False, alpha=0.001, selection='cyclic', tol=0.1))])))
		models.append(('EN', Pipeline([('nor', Normalizer()), ('std', StandardScaler()), ('EN', ElasticNet(normalize=False, alpha=0.001, selection='random', tol=0.2))])))
		models.append(('KNN', KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto')))
		models.append(('CART', Pipeline([('std', StandardScaler()), ('CART', DecisionTreeRegressor(max_features=None, splitter='best', criterion='mse', max_depth=None))])))
		models.append(('SVR', SVR(kernel='rbf', C=1.0)))
		if(ens):
			models.append(('AB', Pipeline([('std', StandardScaler()), ('AB', AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=1000, learning_rate=0.1, loss='linear'))])))
			models.append(('GBM', Pipeline([('std', StandardScaler()), ('GBM', GradientBoostingRegressor(loss='ls', learning_rate=1.0, n_estimators=500, max_features='log2'))])))
			models.append(('RF', Pipeline([('std', StandardScaler()), ('RF', RandomForestRegressor(n_jobs=-1, criterion='mse', max_features='log2', n_estimators=20, max_depth=100))])))
			models.append(('ET', Pipeline([('std', StandardScaler()), ('ET', ExtraTreesRegressor(n_jobs=-1, criterion='mse', max_features=None, n_estimators=20, max_depth=50))])))
	if(clas):		# Classification
		models.append(('LoR', Pipeline([('std', StandardScaler()), ('LoR', LogisticRegression(multi_class='multinomial', C=0.9, solver='sag'))])))
		models.append(('MLP', Pipeline([('nor', Normalizer()), ('std', StandardScaler()), ('MLP', MLPClassifier(alpha=0.001, learning_rate='invscaling', solver='lbfgs'))])))
		models.append(('LDA', Pipeline([('std', StandardScaler()), ('LDA', LinearDiscriminantAnalysis(solver='eigen'))])))
		models.append(('KNN', Pipeline([('std', StandardScaler()), ('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto'))])))
		models.append(('CART', DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=50)))
		models.append(('NB', Pipeline([('std', StandardScaler()), ('NB', GaussianNB())])))
		models.append(('SVM', SVC(kernel='rbf', C=14.0)))
		if(ens):
			models.append(('AB', AdaBoostClassifier(DecisionTreeClassifier(max_depth=None), n_estimators=1000, learning_rate=0.1, algorithm='SAMME')))
			models.append(('GBM', Pipeline([('std', StandardScaler()), ('GBM', GradientBoostingClassifier(max_features='sqrt', loss='deviance', learning_rate=0.1, n_estimators=100))])))
			models.append(('RF', Pipeline([('std', StandardScaler()), ('RF', RandomForestClassifier(max_features='log2', n_estimators=20, criterion='gini', max_depth=50, class_weight=None))])))
			models.append(('ET', Pipeline([('std', StandardScaler()), ('ET', ExtraTreesClassifier(max_features='sqrt', n_estimators=20, criterion='gini', max_depth=100, class_weight=None))])))
	return models


# Evaluate Model
def evaluateAlgorithm(models, num_folds, seed, X_train, Y_train):
	# evaluate each model
	msg = "\n%s\t%s\t%s\n" % ("model", "Score", "desviation")
	results = []
	names = []
	for name, model in models:
		kfold = KFold(n_splits=num_folds, random_state=seed)
		try:
			cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
		except Exception as e:
			cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="r2")
		results.append(cv_results)
		names.append(name)
		msg = msg + "%s: \t%f\t(%f)\n" % (name, cv_results.mean(), cv_results.std())
	return results, names, msg


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
		pipelines.append(('std' + name, Pipeline([('std', StandardScaler()), (name, model)])))
	return pipelines


# Normalize the dataset
def normalizeData(models):
	pipelines = []
	for name, model in models:
		pipelines.append(('nor' + name, Pipeline([('nor', Normalizer()), (name, model)])))
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


# Show accuracy score on Classification
def modelClassificationScore(models, X_train, Y_train, X_test, Y_test, id=""):
	columns = ["Modelo", "Prediccion (%)", "errorMedio (m.)"]
	results = []
	print("%s\t%s\t%s") % (columns[0], columns[1], columns[2])
	for name, model in models:
		model.fit(X_train, Y_train)
		Y_pred = model.predict(X_test)
		acc = sum([abs(ax - bx) <= 0.5 for ax, bx in zip(Y_test, Y_pred)]) / float(len(X_test)) * 100.0
		err = sum([((3 * (ax % 3 - bx % 3) - 1)**2 + (3 * (ax / 3 - bx / 3) - 1)**2)**0.5 for ax, bx in zip(Y_test, Y_pred)]) / float(len(X_test))
		results.append([name, acc, err])
		print(name + "\t" + str(acc) + "\t" + str(err))
	DataFrame(array(results)).to_csv("resultados" + str(id) + ".csv", index=False, header=columns)


# Show accuracy score on Regression
def modelRegressionScore(models, X_train, Y_train, X_test, Y_test, id=""):
	columns = ["Modelo", "Prediccion (%)", "errorMedio (m.)"]
	results = []
	print("%s\t%s\t%s") % (columns[0], columns[1], columns[2])
	for name, model in models:
		model.fit(X_train, ((Y_train - 1) % 3 + 1))
		x_map = model.predict(X_test)
		model.fit(X_train, ((Y_train - 1) / 3))
		y_map = model.predict(X_test)
		Y_pred = array([a + b * 3 for a, b in zip(ceil(x_map - 0.5), ceil(y_map - 0.5))])
		acc = sum([abs(ax - bx) <= 0.5 for ax, bx in zip(Y_test, Y_pred)]) / float(len(X_test)) * 100.0
		err = sum([((ax - bx)**2 + (ay - by)**2)**0.5 for ax, ay, bx, by in zip(x_map, y_map, (Y_test - 1) % 3 + 1, floor((Y_test - 1) / 3))]) / float(len(X_test))
		results.append([name, acc, err])
		print(name + "\t" + str(acc) + "\t" + str(err))
	DataFrame(array(results)).to_csv("resultados" + str(id) + ".csv", index=False, header=columns)


seed = 7
num_folds = 10

dataset = read_csv("controller/Tx_full.csv")
X_train, X_test, Y_train, Y_test = prepareData(dataset, seed)
# take models
models = modelFull(clas=True, reg=False, ens=True)
results, names, msg = evaluateAlgorithm(models, num_folds, seed, X_train, Y_train)
modelClassificationScore(models, X_train, Y_train, X_test, Y_test, "clas")
models = modelFull(clas=False, reg=True, ens=True)
modelRegressionScore(models, X_train, Y_train, X_test, Y_test, "reg")



for i in range(1, 8):
	dataset = read_csv("movil/Tx_0x0" + str(i) + ".csv")
	#plt = dataset.plot(kind='density', subplots=True, layout=(2, 3), sharex=False, sharey=True, title='Smartphone Tx=0x0' + str(i), legend=True)
	#plt[0][0].set_xlabel('Frecuency dBm\n SubFigura a')
	#plt[0][1].set_xlabel('Frecuency dBm\n SubFigura b')
	#plt[0][2].set_xlabel('Frecuency dBm\n SubFigura c')
	#plt[1][0].set_xlabel('Frecuency dBm\n SubFigura d')
	#plt[1][1].set_xlabel('Frecuency dBm\n SubFigura e')
	#plt[1][2].set_xlabel('Sector       \n SubFigura f')
	#fig = pyplot.figure()
	#ax = fig.add_subplot(111)
	#cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
	#fig.colorbar(cax)
	## because of correlation
for i in range(1, 8):
	dataset = read_csv("raspberry/Tx_0x0" + str(i) + ".csv")
	X_train, X_test, Y_train, Y_test = prepareData(dataset, seed)
	models = modelFull(clas=True, reg=False, ens=True)
	modelClassificationScore(models, X_train, Y_train, X_test, Y_test, str(i) + "clas")
	models = modelFull(clas=False, reg=True, ens=True)
	modelRegressionScore(models, X_train, Y_train, X_test, Y_test, str(i) + "reg")
pyplot.show()

for i in range(1, 16):
	indices = [x == i for x in Y_test]
	x1 = list(compress(X_test, indices))
	y1 = list(compress(Y_test, indices))
	print("\nPosicion %d" % (i))
	modelClassificationScore(models, X_train, Y_train, x1, y1)

"""
# model
results, names, msg = evaluateAlgorithm(models, num_folds, seed, X_train, Y_train)
print "\n" + (msg)
compareAlgorithms(results, names)
# normalized
results, names, msg = evaluateAlgorithm(normalizeData(models), num_folds, seed, X_train, Y_train)
print "\n" + (msg)
compareAlgorithms(results, names)
# standarized
results, names, msg = evaluateAlgorithm(standarizeData(models), num_folds, seed, X_train, Y_train)
print "\n" + (msg)
compareAlgorithms(results, names)

results, names, msg = evaluateAlgorithm(normalizeData(standarizeData(models)), num_folds, seed, X_train, Y_train)
print "\n" + (msg)
compareAlgorithms(results, names)
"""

dataset = read_csv("movil/Tx_0x01.csv")
X_train, X_test, Y_train, Y_test = prepareData(dataset, seed)
# take models
models = modelFull(clas=True, reg=False, ens=True)
results, names, msg = evaluateAlgorithm(models, num_folds, seed, X_train, Y_train)
print(msg)
compareAlgorithms(results, names)
models = modelFull(clas=False, reg=True, ens=True)
results, names, msg = evaluateAlgorithm(models, num_folds, seed, X_train, Y_train)
print(msg)
compareAlgorithms(results, names)