import numpy as np
import pandas as pd
import time

import gc, sys
gc.enable()

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

def preproc(df):
    df.rename(columns={'Unnamed: 0':'indx'}, inplace=True)
    # Get patient_id and time of recording
    df['sec'], df['patient_id'] = df['indx'].str.split('.', 1).str
    df.drop(columns=['indx'], inplace=True)
    # Replace id with the corresponding numbers
    patient_map = dict(zip(df['patient_id'].unique(), list(range(1, 501))))
    df = df.replace({'patient_id': patient_map})
    df['sec'] = df['sec'].str[1:]
    df['sec'] = df['sec'].astype('int16')
    # reorder columns
    cols = ['patient_id', 'sec'] + list(df.columns[:-2])
    df = df[cols]
    # sort data by patient_id and time
    df.sort_values(['patient_id', 'sec'], inplace=True)
    # binarize target
    df['y'].loc[df['y'] > 1] = 0
    return df

def feature_extract(data):
    
    X = data.drop(['y', 'sec'], axis=1)
    columns = X.columns[1:]  # drop 'patient_id' from agg
    # The maximum, minimum amplitude and difference btw them:
    max_val_df = X.groupby(['patient_id'])[columns].max().reset_index()
    min_val_df = X.groupby(['patient_id'])[columns].min().reset_index()
    diff = max_val_df - min_val_df
    diff['patient_id'] = max_val_df['patient_id']  # correct patient_id's
    for col in max_val_df.columns[1:]:
        max_val_df.rename(columns={col:col+'_max'}, inplace=True)
    for col in min_val_df.columns[1:]:
        min_val_df.rename(columns={col:col+'_min'}, inplace=True)
    for col in diff.columns[1:]:
        diff.rename(columns={col:col+'_diff'}, inplace=True)
        
    new_features = max_val_df.merge(min_val_df, on=['patient_id'])
    new_features = new_features.merge(diff, on=['patient_id'])
    
    del max_val_df, min_val_df, diff
    gc.collect()
    
    # Sum of positive signal amplitude values:
    p_sum = X.groupby(['patient_id'])[columns].agg(lambda x : x[x > 0].sum()).reset_index()
    for col in p_sum.columns[1:]:
        p_sum.rename(columns={col:col+'_p_sum'}, inplace=True)
        
    new_features = new_features.merge(p_sum, on=['patient_id'])

    del p_sum
    gc.collect()
    
    # RMS signal strength:
    v_rms = X.groupby(['patient_id'])[columns].agg(
            lambda x : np.sqrt((x**2).sum()/23)
        ).reset_index()
    v_rms.fillna(0, inplace=True)
    for col in v_rms.columns[1:]:
        v_rms.rename(columns={col:col+'_rms'}, inplace=True)
        
    new_features = new_features.merge(v_rms, on=['patient_id'])

    del v_rms
    gc.collect()

    df_nf = data.merge(new_features, how='inner', on=['patient_id'])
    cols_order = data.columns[:-1].tolist()+ \
                    new_features.columns[1:].tolist()+ \
                    list(data.columns[-1])
    df_nf = df_nf[cols_order]

    del new_features, X
    gc.collect()
            
    return df_nf

data = reduce_mem_usage(pd.read_csv('../input/epileptic_seizure_recognition_data.csv'))
df = preproc(data)
final_df = feature_extract(df)

final_df.to_csv('../input/data_preproc.csv', index=False)
