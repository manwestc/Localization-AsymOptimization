# Hello World Classification: Iris flowers prediction

# Prepare Problem

# Load libraries
from math import sqrt
from math import ceil
from pandas import read_csv
from pandas.tools.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
# Load dataset
filename = 'iris.data.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# filename = 'pima-indians-diabetes.data.csv'
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = read_csv(filename, names=names)

# Summarize Data

# Descriptive statistics
# shape
print(dataset.shape)
# head
print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('class').size())

# Data visualizations

# box and whisker plots
layout = (int(ceil(sqrt(dataset.shape[1]))), int(round(sqrt(dataset.shape[1]))))
dataset.plot(kind='box', subplots=True, layout=layout, sharex=False, sharey=False)
# pyplot.show()
# histograms
dataset.hist()
# pyplot.show()
# scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

# Prepare Data

# Split-out validation dataset
array = dataset.values
X = array[:, 0:dataset.shape[1] - 1]
Y = array[:, dataset.shape[1] - 1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

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
models = []
models.append(('LR', LogisticRegression()))
models.append(('MLP', MLPClassifier(solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = KFold(n_splits=10, random_state=seed)
	estimators.append((name, model))
	modelPipeline = Pipeline(estimators)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	estimators.pop()
	results.append(cv_results)
	names.append(name)
	msg = "%s: \t%f%%\t(%f%%)" % (name, cv_results.mean() * 100.0, cv_results.std() * 100.0)
	print(msg)

# Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
# pyplot.boxplot(results)
ax.set_xticklabels(names)
# pyplot.show()

# Make predictions on validation dataset
print("Test Validacion")
knn = DecisionTreeClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(knn.feature_importances_)
