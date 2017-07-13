import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import time, datetime

# ['date_account_created', 'timestamp_first_active', 'gender', 'age',
# 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider',
# 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']


# Loading data
df_train = pd.read_csv("./abbData/train_users.csv")
df_test = pd.read_csv("./abbData/test_users.csv")
labels = df_train['country_destination'].values
# df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0] # of training data

# Creating a DataFrame with train and test data for encoding etc.
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking','country_destination'], axis=1)


## ===== feature engineering =======
# time series: 'date_account_created' date to datestamps
df_all['created']= map(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()), df_all['date_account_created'])
df_all = df_all.drop(['date_account_created'], axis = 1)
# Plot to see if season and trend.
# df_1 = df_all[df_all['country_destination']=='US']
# plt.hist(df_1['created'],bins = 50)

# time series: 'timestamp_first_active'
df_all['first_active'] = map(lambda x: time.mktime(datetime.datetime.strptime(str(x), "%Y%m%d%H%M%S").timetuple()), df_all['timestamp_first_active'])
df_all = df_all.drop(['timestamp_first_active'], axis=1)

# Age: missing values, unreasonal values
# df_all.isnull().sum() #check how many missing data
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>100), np.nan, av)
imr = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
df_age = df_all['age'].reshape(len(df_all['age']), -1)
imr = imr.fit(df_age)
df_all['age'] = imr.transform(pd.DataFrame(df_age).values)

# 'first_affiliate_tracked': missing data
# df_all['first_affiliate_tracked'].value_counts()
df_all['first_affiliate_tracked'] = df_all['first_affiliate_tracked'].replace(to_replace=np.nan, value='untracked',regex = True)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)

#Splitting train and test
vals = df_all.values
X = vals[:piv_train,:]
X_test = vals[piv_train:,:]
le = LabelEncoder()
y = le.fit_transform(labels)

# standardize and scaling:
# sc = StandardScaler()
# sc.fit(X)
# Xtr = sc.transform(X)
# Xte = sc.transform(X_test)
Xtr = X

# try xgboost
from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
xgb.fit(Xtr, y)
y_pred = xgb.predict(Xtr)
cm = confusion_matrix(y, y_pred)
score = accuracy_score(y, y_pred)
print(score, cm)