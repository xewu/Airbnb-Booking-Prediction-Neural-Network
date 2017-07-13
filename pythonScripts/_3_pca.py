import numpy as np
import pandas as pd


X_train_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/X_train.csv'
X_valid_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/X_valid.csv'
X_test_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/X_test.csv'
X_tmp_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/X_tmp.csv'
X_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/X.csv'

y_train_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/y_train.csv'
y_valid_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/y_valid.csv'
y_test_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/y_test.csv'
y_tmp_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/y_tmp.csv'
y_path = '/Users/Erica/PycharmProjects/208_project/feature_engineering_input/y.csv'

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
X_valid = pd.read_csv(X_valid_path)
X_tmp = pd.read_csv(X_tmp_path)
X = pd.read_csv(X_path)

y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)
y_valid = pd.read_csv(y_valid_path)
y_tmp = pd.read_csv(y_tmp_path)
y = pd.read_csv(y_path)

from sklearn import decomposition
import matplotlib.pyplot as plt
pca = decomposition.PCA()
pca.fit(X)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')
plt.xlim(xmax=10)

# from plot n=4
n_components = 4
pca = decomposition.PCA(n_components=n_components)
X_pca = pca.fit(X).transform(X) # shape = (213450, 4)

# np.savetxt('X_pca.csv', X_pca, delimiter=',')