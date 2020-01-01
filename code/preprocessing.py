import gc
gc.enable()

from datetime import datetime

import numpy as np
import pandas as pd
import config

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.decomposition import PCA
from rootClass import rootClass


class DataPreprocessor(rootClass):
    """"
    Класс для очистки и предобработки данных
    """

    def __init__(self):
        rootClass.__init__(self)

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            res = func(*args, **kwargs)
            print(f'{func.__name__} done! Time:', datetime.now() - start)
            return res
        return wrapper

    def check_missing_values(self, df):
        """
        calculate missing values by columns
        :param df: dataframe
        :return dataframe with missing information
        """
        try:
            # Total missing values
            mis_val = df.isnull().sum()
            # Percentage of missing values
            mis_val_percent = 100 * df.isnull().sum() / len(df)
            # Make a table with the results
            mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
            # Rename the columns
            mis_val_table_ren_columns = mis_val_table.rename(
                columns={0: 'Missing Values', 1: '% of Total Values'})
            # Sort the table by percentage of missing descending
            mis_val_table_ren_columns = mis_val_table_ren_columns[
                mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
                '% of Total Values', ascending=False).round(1)
            # Print some summary information
            print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                      "There are " + str(
                mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")
        except Exception:
            self.print_exception_message()
        return mis_val_table_ren_columns

    @timeit
    def data_preparation(self, df):
        """
        create 'patient_id' and 'sec' columns
        :param df: dataframe
        :return dataframe
        """
        print('start preparation...')
        try:
            df.reset_index(inplace=True)
            df.rename(columns={'Unnamed: 0': 'indx'}, inplace=True)
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
            df_pr = df[cols]
            # sort data by patient_id and time
            df_pr.sort_values(['patient_id', 'sec'], inplace=True)
            # binarize target
            df_pr['y'].loc[df_pr['y'] > 1] = 0
        except Exception:
            self.print_exception_message()
        return df_pr

    @timeit
    def feature_extract(self, df_pr):
        """
        create new features
        :param df_pr: dataframe after preparation
        :return dataframe with new features
        """
        print('create new features...')
        try:
            X = df_pr.drop(['y', 'sec'], axis=1)
            columns = X.columns[1:]  # drop 'patient_id' from agg
            # The maximum, minimum amplitude and difference btw them:
            max_val_df = X.groupby(['patient_id'])[columns].max().reset_index()
            min_val_df = X.groupby(['patient_id'])[columns].min().reset_index()
            diff = max_val_df - min_val_df
            diff['patient_id'] = max_val_df['patient_id']  # correct patient_id's
            for col in max_val_df.columns[1:]:
                max_val_df.rename(columns={col: col + '_max'}, inplace=True)
            for col in min_val_df.columns[1:]:
                min_val_df.rename(columns={col: col + '_min'}, inplace=True)
            for col in diff.columns[1:]:
                diff.rename(columns={col: col + '_diff'}, inplace=True)

            new_features = max_val_df.merge(min_val_df, on=['patient_id'])
            new_features = new_features.merge(diff, on=['patient_id'])
            del max_val_df, min_val_df, diff
            gc.collect()

            # Sum of positive signal amplitude values:
            p_sum = X.groupby(['patient_id'])[columns].agg(lambda x: x[x > 0].sum()).reset_index()
            for col in p_sum.columns[1:]:
                p_sum.rename(columns={col: col + '_p_sum'}, inplace=True)

            new_features = new_features.merge(p_sum, on=['patient_id'])
            del p_sum
            gc.collect()

            # RMS signal strength:
            v_rms = X.groupby(['patient_id'])[columns].agg(
                lambda x: np.sqrt((x ** 2).sum() / 23)
            ).reset_index()
            v_rms.fillna(0, inplace=True)
            for col in v_rms.columns[1:]:
                v_rms.rename(columns={col: col + '_rms'}, inplace=True)

            new_features = new_features.merge(v_rms, on=['patient_id'])
            del v_rms
            gc.collect()
            # add new features to original dataframe
            df_nf = df_pr.merge(new_features, how='inner', on=['patient_id'])
            del new_features, X
            gc.collect()

            print('Old dataframe shape: ', df_pr.shape)
            print('New dataframe shape: ', df_nf.shape)
            print()
        except Exception:
            self.print_exception_message()
        return df_nf

    def select_label_target(self, df_nf, target_column):
        """
        select X and y dataframes by target column
        :param df_nf: dataframe with new features
        :param target_column: target column name
        :return X and y dataframes
        """
        try:
            X = df_nf.drop(columns=[target_column])
            y = df_nf[target_column]
        except Exception:
            self.print_exception_message()
        return X, y

    def scaler(self, scaler_type):
        """
        select data scaler
        :param scaler_type: scaler type
        :return scaler object
        """
        try:
            if scaler_type == config.DATA_STANDARD_SCALER:
                scaler = StandardScaler()
            elif scaler_type == config.DATA_ROBUST_SCALER:
                scaler = RobustScaler()
            elif scaler_type == config.DATA_ROBUST_SCALER:
                scaler = Normalizer()
            elif scaler_type == config.DATA_ROBUST_SCALER:
                scaler = MinMaxScaler()
            elif scaler_type == config.DATA_ROBUST_SCALER:
                scaler = MaxAbsScaler()
        except Exception:
            self.print_exception_message()
        return scaler

    @timeit
    def reduce_dimension(self, n):
        """
        initialize PCA for pipeline
        :param n: n-components for PCA
        :return PCA object
        """
        pca = PCA(n_components=n)
        return pca

    def split_data(self, X, y, test_size_percentage, random_state, stratify_target=None):
        """
        split X and y to train and test data
        :param X: X dataframe
        :param y: y dataframe
        :param test_size_percentage: test size in percentage (%)
        :param random_state: random state initial value
        :param stratify_target: used stratify target
        :return: generator object with train-test indexes
        """
        try:
            if stratify_target is None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage / 100,
                                                                    random_state=random_state)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage / 100,
                                                                    stratify=y, random_state=random_state)
        except Exception:
            self.print_exception_message()
        return X_train, X_test, y_train, y_test

    def cv_folds(self, X_train, y_train, splits):
        """
        split X_train and y_train for evaluation on cross-validate
        :param X_train: train dataframe
        :param y_train: train targets
        :param splits: n-folds for cross-validate
        return: generator object with train-test indexes
        """
        try:
            groups = X_train['patient_id'].values
            gkf = GroupKFold(n_splits=splits).split(X_train, y_train, groups)
        except Exception:
            self.print_exception_message()
        return gkf
