import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

X_pca_path = '/Users/Erica/PycharmProjects/208_project/X_pca.csv'
y_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/y.csv'

X = pd.read_csv(X_pca_path, header=None).values
y = pd.read_csv(y_path, header=None).values
y = y.ravel()

X1 = X[y==7, :]
y1 = y[y==7]
X2 = X[y==10, :]
y2 = y[y==10]

X = np.concatenate((X1,X2), axis=0)
y = np.concatenate((y1,y2), axis=0)

n_classes = 12
random_state = 1

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

GSW = [(0,107,182), (253,185,39)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# Generally take a look at how the observations distributed, feature by feature.
cm = plt.cm.RdBu
cm_bright = ListedColormap([tableau20[2], tableau20[8]])

names = ["Logistic", "Nearest Neighbors", "Classification Tree", "GBC",
         "Random Forest", "AdaBoost",
         "Quadratic Discriminant Analysis",
         "Linear Discriminant Analysis"]

classifiers = [
    LogisticRegression(random_state=random_state),
    KNeighborsClassifier(n_neighbors=100),
    DecisionTreeClassifier(max_depth=5),
    GradientBoostingClassifier(n_estimators=50,
                               random_state=random_state),
    RandomForestClassifier(max_depth=5, n_estimators=20),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis()]



X = preprocessing.scale(X)
h = 0.02 # step size in mesh
figure = plt.figure(figsize=(66, 11))

a = X[:, [1,2]]
b = X[:, [1,3]]
c = X[:, [2,3]]
d = X[:, [0,1]]
e = X[:, [0,2]]
f = X[:, [0,3]]
datasets = [d,e,f,a,b,c]




j=1
for i in range(len(datasets)):
    X = datasets[i]
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=.4)

    # do not need to plot all points, just rancom choose 100 training and 30 testing to pot in figure
    line_sample = random.sample(range(1, len(X_train)), 100)
    Xs = X_train[line_sample, :]
    ys = y_train[line_sample]

    line_sample_te = random.sample(range(1, len(X_test)), 30)
    Xs_test = X_test[line_sample_te, :]
    ys_test = y_test[line_sample_te]

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    ax = plt.subplot(len(datasets), len(classifiers) + 1, j)
    # Plot the training points
    ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=cm_bright)
    # and testing points
    ax.scatter(Xs_test[:, 0], Xs_test[:, 1], c=ys_test, cmap=cm_bright, alpha=0.5)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    j += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, j)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot also the training points
        ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap=cm_bright)
        # and testing points
        ax.scatter(Xs_test[:, 0], Xs_test[:, 1], c=ys_test, cmap=cm_bright,
                   alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        j += 1

figure.subplots_adjust(left=.02, right=.98)
plt.show()