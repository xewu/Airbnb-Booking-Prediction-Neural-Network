import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis



import _0_featureEngineering as fe
X_train, X_valid, X_test, X_tmp = fe.X_train, fe.X_valid, fe.X_test, fe.X_tmp
y_train, y_valid, y_test, y_tmp = fe.y_train, fe.y_valid, fe.y_test, fe.y_tmp

import _1_minLogLoss as em
EN_optA = em.EN_optA
EN_optB = em.EN_optB


# classification classes
n_classes = 12
#fixing random state
random_state=1

#Defining the classifiers
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

#predictions on the validation and test sets
p_valid = []
p_test = []

print('Performance of individual classifiers (1st layer) on X_test')
print('------------------------------------------------------------')

for nm, clf in zip(names, classifiers):
    #First run. Training on (X_train, y_train) and predicting on X_valid.
    clf.fit(X_train, y_train)
    yv = clf.predict_proba(X_valid)
    p_valid.append(yv)

    #Second run. Training on (X, y) and predicting on X_test.
    clf.fit(X_tmp, y_tmp)
    yt = clf.predict_proba(X_test)
    yhat = clf.predict(X_test)
    p_test.append(yt)

    print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(nm), 'logloss  =>', log_loss(y_test, yt)))
    print('        accuracy score  =>', accuracy_score(y_test, yhat))
    print('')
    #Printing out the performance of the classifier
#    print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(nm), 'logloss  =>', log_loss(y_test, yt),
#                                        'accuracy score  =>', accuracy_score(y_test, yt)))
#Printing out the performance of the classifier

# ================================ Second layer ===========================

print('Performance of optimization based ensemblers (2nd layer) on X_test')
print('------------------------------------------------------------')

#Creating the data for the 2nd layer.
XV = np.hstack(p_valid)
XT = np.hstack(p_test)

#EN_optA
enA = EN_optA(n_classes)
enA.fit(XV, y_valid)
w_enA = enA.w
y_enA = enA.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('EN_optA:', 'logloss  =>', log_loss(y_test, y_enA)))

#Calibrated version of EN_optA
cc_optA = CalibratedClassifierCV(enA, method='isotonic')
cc_optA.fit(XV, y_valid)
y_ccA = cc_optA.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optA:', 'logloss  =>', log_loss(y_test, y_ccA)))

#EN_optB
enB = EN_optB(n_classes)
enB.fit(XV, y_valid)
w_enB = enB.w
y_enB = enB.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('EN_optB:', 'logloss  =>', log_loss(y_test, y_enB)))

#Calibrated version of EN_optB
cc_optB = CalibratedClassifierCV(enB, method='isotonic')
cc_optB.fit(XV, y_valid)
y_ccB = cc_optB.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB)))
print('')