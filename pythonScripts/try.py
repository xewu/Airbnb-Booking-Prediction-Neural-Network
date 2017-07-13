import numpy as np
import pandas as pd
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
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)

X_pca_path = '/Users/Erica/PycharmProjects/208_project/X_pca.csv'
y_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/y.csv'

X = pd.read_csv(X_pca_path, header=None).values
y = pd.read_csv(y_path, header=None).values
y = y.ravel()

X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2)

n_classes = 12
random_state = 1

names = ["Logistic", "Nearest Neighbors", "XGBoost", "GBC",
         "Random Forest", "AdaBoost",
         "Quadratic Discriminant Analysis",
         "Linear Discriminant Analysis"]

classifiers = [
    LogisticRegression(random_state=random_state),
    KNeighborsClassifier(n_neighbors=100),
    XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                  objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0),
    GradientBoostingClassifier(n_estimators=50,
                               random_state=random_state),
    RandomForestClassifier(max_depth=5, n_estimators=20),
    AdaBoostClassifier(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis()]

p_Xtr = []
p_test1 = []
print('Individual classifiers (1st layer)')
print('------------------------------------------------------------')


for nm, clf in zip(names, classifiers):
#First run. Training on (X_train, y_train) and predicting on X_valid.
    clf.fit(X_train, y_train)
    ytr = clf.predict_proba(X_train)
    yhd = clf.predict_proba(X_hold)
    yhat = clf.predict(X_train)
    p_Xtr.append(ytr)
    p_test1.append(yhd)

#Printing out the performance of the classifier
    print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(nm), 'logloss  =>', log_loss(y_train, ytr)))
    print('{:10s} {:2s} {:1.7f}'.format('%s: ' % (nm), 'accuracy  =>', accuracy_score(y_train, yhat)))
    print('')

# ============================================================================
# ============================================================================
print('============================================================================')
print('combination classifiers (2st layer)')
print('------------------------------------------------------------')




def objf_ens_optA(w, Xs, y, n_class=12):
    """
    Function to be minimized in the EN_optA ensembler.

    Parameters:
    ----------
    w: array-like, shape=(n_preds)
       Candidate solution to the optimization problem (vector of weights).
    Xs: list of predictions to combine
       Each prediction is the solution of an individual classifier and has a
       shape=(n_samples, n_classes).
    y: array-like sahpe=(n_samples,)
       Class labels
    n_class: int
       Number of classes in the problem (12 in Airbnb competition)

    Return:
    ------
    score: Score of the candidate solution.
    """
    w = np.abs(w)
    sol = np.zeros(Xs[0].shape)
    for i in range(len(w)):
        sol += Xs[i] * w[i]
    #Using log-loss as objective function (different objective functions can be used here).
    score = log_loss(y, sol)
    return score


class EN_optA(BaseEstimator):
    """
    Given a set of predictions $X_1, X_2, ..., X_n$,  it computes the optimal set of weights
    $w_1, w_2, ..., w_n$; such that minimizes $log\_loss(y_T, y_E)$,
    where $y_E = X_1*w_1 + X_2*w_2 +...+ X_n*w_n$ and $y_T$ is the true solution.
    """
    def __init__(self, n_class=12):
        super(EN_optA, self).__init__()
        self.n_class = n_class

    def fit(self, X, y):
        """
        Learn the optimal weights by solving an optimization problem.

        Parameters:
        ----------
        Xs: list of predictions to be ensembled
           Each prediction is the solution of an individual classifier and has
           shape=(n_samples, n_classes).
        y: array-like
           Class labels
        """
        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        #Initial solution has equal weight for all individual predictions.
        x0 = np.ones(len(Xs)) / float(len(Xs))
        #Weights must be bounded in [0, 1]
        bounds = [(0,1)]*len(x0)
        #All weights must sum to 1
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        #Calling the solver
        res = minimize(objf_ens_optA, x0, args=(Xs, y, self.n_class),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=cons
                       )
        self.w = res.x
        return self

    def predict_proba(self, X):
        """
        Use the weights learned in training to predict class probabilities.

        Parameters:
        ----------
        Xs: list of predictions to be blended.
            Each prediction is the solution of an individual classifier and has
            shape=(n_samples, n_classes).

        Return:
        ------
        y_pred: array_like, shape=(n_samples, n_class)
                The blended prediction.
        """
        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        y_pred = np.zeros(Xs[0].shape)
        for i in range(len(self.w)):
            y_pred += Xs[i] * self.w[i]
        return y_pred

def objf_ens_optB(w, Xs, y, n_class=12):
    """
    Function to be minimized in the EN_optB ensembler.

    Parameters:
    ----------
    w: array-like, shape=(n_preds)
       Candidate solution to the optimization problem (vector of weights).
    Xs: list of predictions to combine
       Each prediction is the solution of an individual classifier and has a
       shape=(n_samples, n_classes).
    y: array-like sahpe=(n_samples,)
       Class labels
    n_class: int
       Number of classes in the problem, i.e. = 12

    Return:
    ------
    score: Score of the candidate solution.
    """
    #Constraining the weights for each class to sum up to 1.
    #This constraint can be defined in the scipy.minimize function, but doing
    #it here gives more flexibility to the scipy.minimize function
    #(e.g. more solvers are allowed).
    w_range = np.arange(len(w))%n_class
    for i in range(n_class):
        w[w_range==i] = w[w_range==i] / np.sum(w[w_range==i])

    sol = np.zeros(Xs[0].shape)
    for i in range(len(w)):
        sol[:, i % n_class] += Xs[int(i / n_class)][:, i % n_class] * w[i]

    #Using log-loss as objective function (different objective functions can be used here).
    score = log_loss(y, sol)
    return score



#####
class EN_optB(BaseEstimator):
    """
    Given a set of predictions $X_1, X_2, ..., X_n$, where each $X_i$ has
    $m=12$ clases, i.e. $X_i = X_{i1}, X_{i2},...,X_{im}$. The algorithm finds the optimal
    set of weights $w_{11}, w_{12}, ..., w_{nm}$; such that minimizes
    $log\_loss(y_T, y_E)$, where $y_E = X_{11}*w_{11} +... + X_{21}*w_{21} + ...
    + X_{nm}*w_{nm}$ and and $y_T$ is the true solution.
    """
    def __init__(self, n_class=12):
        super(EN_optB, self).__init__()
        self.n_class = n_class

    def fit(self, X, y):
        """
        Learn the optimal weights by solving an optimization problem.

        Parameters:
        ----------
        Xs: list of predictions to be ensembled
           Each prediction is the solution of an individual classifier and has
           shape=(n_samples, n_classes).
        y: array-like
           Class labels
        """
        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        #Initial solution has equal weight for all individual predictions.
        x0 = np.ones(self.n_class * len(Xs)) / float(len(Xs))
        #Weights must be bounded in [0, 1]
        bounds = [(0,1)]*len(x0)
        #Calling the solver (constraints are directly defined in the objective
        #function)
        res = minimize(objf_ens_optB, x0, args=(Xs, y, self.n_class),
                       method='L-BFGS-B',
                       bounds=bounds
                       )
        self.w = res.x
        return self

    def predict_proba(self, X):
        """
        Use the weights learned in training to predict class probabilities.

        Parameters:
        ----------
        Xs: list of predictions to be ensembled
            Each prediction is the solution of an individual classifier and has
            shape=(n_samples, n_classes).

        Return:
        ------
        y_pred: array_like, shape=(n_samples, n_class)
                The ensembled prediction.
        """
        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        y_pred = np.zeros(Xs[0].shape)
        for i in range(len(self.w)):
            y_pred[:, i % self.n_class] += \
                   Xs[int(i / int(self.n_class))][:, i % self.n_class] * self.w[i]
        return y_pred

# Layer Computing:
Xtr = np.hstack(p_Xtr)
X_hold2 = np.hstack(p_test1)

opt_names = ["SLSQP","Calibrated-SLSQP", "SGD", "XGB"]
enA = EN_optA(n_classes)
sgd = SGDClassifier(loss="log", alpha=0.01, n_iter=200)
XGB = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
              objective='multi:softprob', subsample=0.5,
              colsample_bytree=0.5, seed=0)
optimizers = [enA,
              CalibratedClassifierCV(enA, method='isotonic'),
              sgd,XGB]

p_Xtr2 = []
p_test2 = []

for name, opt in zip(opt_names, optimizers):
#First run. Training on (X_train, y_train) and predicting on X_valid.
    opt.fit(Xtr, y_train)
    yt = opt.predict_proba(Xtr)
    yhd = opt.predict_proba(X_hold2)
    p_Xtr2.append(yt)
    p_test2.append(yhd)
#Printing out the performance of the classifier
    print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(name), 'logloss  =>', log_loss(y_train, yt)))
    print('')


print('============================================================================')
print('Stochastic Gradient Descent (3rd output layer)')
print('------------------------------------------------------------')
X_tr3 = np.hstack(p_Xtr2)
X_hold3 = np.hstack(p_test2)

sgd.fit(X_tr3, y_train)
y_sgd = sgd.predict_proba(X_tr3)
print('{:20s} {:2s} {:1.7f}'.format('Stochastic Gradient Descenting:', 'logloss  =>', log_loss(y_train, y_sgd)))
print('')

XGB.fit(X_tr3, y_train)
y_XGB = XGB.predict_proba(X_tr3)
y_hat = XGB.predict(X_tr3)
print('{:20s} {:2s} {:1.7f}'.format('XGB:', 'logloss  =>', log_loss(y_train, y_XGB)))
print('')

# ==========

def accuracy_rate(y_pro, cut):
    y_sorted = []
    for i in range(len(y_pro)):
        tmp = y_pro[i, :].argsort()[::-1]
        y_sorted.append(tmp)
    y_get_all = np.asarray(y_sorted)
    y_getn = y_get_all[:, 0:cut]
    count_correct = 0
    for i in range(len(y_train)):
        if y_train[i] in y_getn[i]:
            count_correct += 1
    rate = float(count_correct) / len(y_train)
    return rate

y_pro = y_XGB
rate1 = accuracy_rate(y_pro,cut=1)
print('{:2s} {:1.7f}'.format('1 rm accurate rate =>', rate1))
print('')
rate2 = accuracy_rate(y_pro,cut=2)
print('{:2s} {:1.7f}'.format('2 rm accurate rate =>', rate2))
print('')
rate3 = accuracy_rate(y_pro,cut=3)
print('{:2s} {:1.7f}'.format('3 rm accurate rate =>', rate3))
print('')
