import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


# Loading data
df_train = pd.read_csv("./abbData/train_users.csv")
df_test = pd.read_csv("./abbData/test_users.csv")
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0] # of training data

# Creating a DataFrame with train and test data for encoding etc.
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
# Filling nan
df_all = df_all.fillna(-1)

