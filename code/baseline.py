import numpy as np
import pandas as pd
import time

import gc, sys
gc.enable()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline



# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def preproc(data: pd.DataFrame()):
    y = data['y'].values
    X = data.drop(['y', 'patient_id', 'sec'], axis=1).values
    groups = data['patient_id'].values
    return X, y, groups

#-----------------------------------------------------------------------------------
df = reduce_mem_usage(pd.read_csv('../input/data_preproc.csv'))
X, y, groups = preproc(df)

# create pipeline
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('pca', PCA(n_components=50)),
    ('clf', SVC(kernel='rbf', gamma='auto', 
                C=1, class_weight='balanced'))])


gkf = GroupKFold(n_splits=10).split(X, y, groups)
scoring = 'roc_auc'
# evaluate pipeline
score = cross_val_score(pipe, X, y, cv=gkf, scoring=scoring, n_jobs=-1)
print('Mean ROC_AUC: ', np.mean(score))
print('ROC_AUC std: ', np.std(score))
