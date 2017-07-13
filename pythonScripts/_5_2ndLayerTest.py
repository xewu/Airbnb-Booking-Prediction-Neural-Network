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
# cc_optB = CalibratedClassifierCV(enB, method='isotonic')
# cc_optB.fit(XV, y_valid)
# y_ccB = cc_optB.predict_proba(XT)
# print('{:20s} {:2s} {:1.7f}'.format('Calibrated_EN_optB:', 'logloss  =>', log_loss(y_test, y_ccB)))
# print('')

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss="log", alpha=0.01, n_iter=200)
sgd.fit(XV, y_valid)
y_sgd = sgd.predict_proba(XT)
print('{:20s} {:2s} {:1.7f}'.format('Stochastic Gradient Descenting:', 'logloss  =>', log_loss(y_test, y_sgd)))
print('')