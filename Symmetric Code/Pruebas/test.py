import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#test
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
from sklearn.datasets import load_iris
from pandas import read_csv

h = .02  # step size in the mesh

names = [#"Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         #"Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         #"Naive Bayes", "QDA", "LogReg"
         "LDA", "Regresion Logistica", "Neural Network",
         "LDA", "Regresion Logistica", "Neural Network",
         "LDA", "Regresion Logistica", "Neural Network",
         "LDA", "Regresion Logistica"
         ]

classifiers = [
    Pipeline([('std', StandardScaler()), ('LoR', LogisticRegression(multi_class='multinomial', C=0.9, solver='sag'))]),
    Pipeline([('nor', Normalizer()), ('std', StandardScaler()), ('MLP', MLPClassifier(alpha=0.001, learning_rate='invscaling', solver='lbfgs'))]),
    Pipeline([('std', StandardScaler()), ('LDA', LinearDiscriminantAnalysis(solver='eigen'))]),
    Pipeline([('std', StandardScaler()), ('KNN', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto'))]),
    DecisionTreeClassifier(splitter='best', criterion='entropy', max_depth=50),
    Pipeline([('std', StandardScaler()), ('NB', GaussianNB())]),
    SVC(kernel='rbf', C=14.0),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=None), n_estimators=1000, learning_rate=0.1, algorithm='SAMME'),
    Pipeline([('std', StandardScaler()), ('GBM', GradientBoostingClassifier(max_features='sqrt', loss='deviance', learning_rate=0.1, n_estimators=100))]),
    Pipeline([('std', StandardScaler()), ('RF', RandomForestClassifier(max_features='log2', n_estimators=20, criterion='gini', max_depth=50, class_weight=None))]),
    Pipeline([('std', StandardScaler()), ('ET', ExtraTreesClassifier(max_features='sqrt', n_estimators=20, criterion='gini', max_depth=100, class_weight=None))])
]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=7, n_clusters_per_class=1)
rng = np.random.RandomState(7)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
dtTx = read_csv("info3.csv")
datasets = [#make_moons(noise=0.3, random_state=0),
            #make_circles(noise=0.2, factor=0.5, random_state=1),
            #(np.array(load_iris().data[:150, 0:2]), np.array(load_iris().target[0:150])),
            (dtTx.values[:, 0:2], dtTx.values[:,-1]),
            (np.array(load_iris().data[:150, 2:4]), np.array(load_iris().target[0:150]))
            ]
dataImagen = ["Sepal wl","Petal wl"]
figure = plt.figure(figsize=(27, 9))
j = 0
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.3, random_state=7)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dtTx first
    cm = plt.cm.Spectral
    cm_bright = ListedColormap(['#FF0000', '#FFCC00', '#00BBFF'])
    #ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    # and testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.text(xx.max() - .3, yy.min() + .3, dataImagen[j], size=15, horizontalalignment='right')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    j += 1
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        #if hasattr(clf, "decision_function"):
        #    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        #    print("1"+str(name))
        #else:
        #    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        #    print("2"+str(name))

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.6)

        # Plot also the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
