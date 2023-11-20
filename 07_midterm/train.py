import pickle as pkl
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.model_selection import train_test_split,RepeatedStratifiedKFold, cross_val_score

# from sklearn.model_selection import KFold
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score
from sklearn import metrics
# from tqdm.auto import tqdm
# import xgboost as xgb
from xgboost import XGBClassifier


n_estimators = 600
max_depth = 6

df = pd.read_csv("./smoker_train_dataset.csv")
pg = pd.read_csv("./train.csv")

df = pd.concat([pg, df])

df.drop(columns=['id'], inplace=True)

df['bmi'] = df['weight(kg)'] / (df['height(cm)'] / 100) ** 2
features = list(df.columns)
features.pop(-2)

df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(
    df_train_full, test_size=0.25, random_state=42)

full_train = df_train_full.copy()
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train_full = df_train_full.smoking.values
y_train = df_train.smoking.values
y_val = df_val.smoking.values
y_test = df_test.smoking.values

del full_train['smoking']
del df_train['smoking']
del df_val['smoking']
del df_test['smoking']

xgb_model = XGBClassifier(tree_method = 'hist',
                    n_estimators = 600,
                    max_depth = 6,
                    learning_rate = 0.1,
                    colsample_bytree = 0.5)

xgb_model.fit(full_train, y_train_full)
y_pred_prob = xgb_model.predict_proba(df_test)[:, 1]
y_pred = xgb_model.predict(df_test)
roc = metrics.roc_auc_score(y_test, y_pred_prob)
acc = metrics.accuracy_score(y_test, y_pred)

output_file = f'./model_xgboost.bin'

with open(output_file,'wb') as f_out:
    pkl.dump(xgb_model,f_out)
    
print("----- Model Exported -----")

# test = full_train.tail(1).values
# test_pred = xgb_model.predict(test)
# print(test_pred)