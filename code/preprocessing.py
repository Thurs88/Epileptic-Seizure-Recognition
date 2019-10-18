import os
import sys
import gc

gc.enable()
import traceback
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config

from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.decomposition import PCA


class EpilepticRecognitionClass(object):

    def __init__(self):
        pass


    def read_data(self, csv_file_name, csv_file_path, separator=",", reduce_memory=False, encoding_unicode=None):
        """
        read data from a csv file
        :param file_name: csv file name
        :param csv_file_path: csv file path
        :param separator: column separator
        :reduce_memory: use/not reduce_memory function
        :param encoding_unicode: csv file encoding unicode
        :return dataframe
        """
        df = None
        try:
            if reduce_memory == True:
                df = self.reduce_mem_usage(pd.read_csv(csv_file_path + csv_file_name, sep=separator))
            else:
                df = pd.read_csv(csv_file_path + csv_file_name, sep=separator)
            if encoding_unicode is None:
                df = pd.read_csv(csv_file_path + csv_file_name, sep=separator)
            else:
                df = pd.read_csv(csv_file_path + csv_file_name, sep=separator, encoding=encoding_unicode)
        except Exception:
            self.print_exception_message()
        return df


    def reduce_mem_usage(self, df):
        """
        Memory saving function
        Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
        :param df: original csv file
        :return dataframe
        """
        start_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
        try:
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
        except Exception:
            self.print_exception_message()

        end_mem = df.memory_usage().sum() / 1024 ** 2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
        print()
        return df


    def show_file_information(self, df):
        """
        show data file information
        :param df: dataframe
        :return none
        """
        try:
            print("------------DATASET INFORMATION------------")
            df.info()
            print('-------------------------------------------')
            print()
        except Exception:
            self.print_exception_message()


    def show_file_data(self, df):
        """
        print data file
        :param df: dataframe
        :return none
        """
        try:
            print("----------------DATA SAMPLE----------------")
            print(df.head())
            print('Dataframe shape:', df.shape)
            print('-------------------------------------------')
            print()
        except Exception:
            self.print_exception_message()


    def show_descriptive_statistics(self, df):
        """
        show descriptive statistics for numerical labels (features)
        :param df: dataframe
        :return none
        """
        try:
            print("------------DESCRIPTIVE STATISTICS------------")
            descriptive_statistics = df.describe().T
            print(descriptive_statistics)
            print()
            print('Data types: ', list(set(df.dtypes.tolist())))
            print('Columns with object-type data: ',
                  list(set(df.select_dtypes(include=['O']).columns.tolist())))
            print('-----------------------------------------------')
            print()
        except Exception:
            self.print_exception_message()


    def check_missing_values(self, df):
        """
        calculate missing values by columns
        :param df: dataframe
        :return dataframe with missing information
        """
        print('--------------MISSING VALUES--------------')
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
                  "There are " + str(mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")
        except Exception:
            self.print_exception_message()
        print('-------------------------------------------')
        print()
        return mis_val_table_ren_columns


    def data_preparation(self, df):
        """
        create 'patient_id' and 'sec' columns
        :param df: dataframe
        :return dataframe
        """
        print('start preparation...')
        try:
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


    def data_scaler(self, scaler_type):
        """
        select data scaler
        :param scaler_type: scaler type
        :return scaler object
        """
        try:
            if (scaler_type == config.DATA_STANDARD_SCALER):
                scaler = StandardScaler()
            elif (scaler_type == config.DATA_ROBUST_SCALER):
                scaler = RobustScaler()
            elif (scaler_type == config.DATA_ROBUST_SCALER):
                scaler = Normalizer()
            elif (scaler_type == config.DATA_ROBUST_SCALER):
                scaler = MinMaxScaler()
            elif (scaler_type == config.DATA_ROBUST_SCALER):
                scaler = MaxAbsScaler()
        except Exception:
            self.print_exception_message()
        return scaler


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


    def get_submission(self, out_path, out_name, y_pred):
        """
        read target predictions to csv file
        :param out_path: path to results dir
        :param out_name: results file name
        :param y_pred: predicted y
        :return: none
        """
        out = np.column_stack((range(1, y_pred.shape[0] + 1), y_pred))
        np.savetxt(out_path + out_name, out, header="patient_id, y", comments="", fmt="%d,%d")


    @staticmethod
    def print_exception_message(message_orientation="horizontal"):
        """
        print full exception message
        :param message_orientation: horizontal or vertical
        :return none
        """
        try:
            exc_type, exc_value, exc_tb = sys.exc_info()
            file_name, line_number, procedure_name, line_code = traceback.extract_tb(exc_tb)[-1]
            time_stamp = " [Time Stamp]: " + str(time.strftime("%Y-%m-%d %I:%M:%S %p"))
            file_name = " [File Name]: " + str(file_name)
            procedure_name = " [Procedure Name]: " + str(procedure_name)
            error_message = " [Error Message]: " + str(exc_value)
            error_type = " [Error Type]: " + str(exc_type)
            line_number = " [Line Number]: " + str(line_number)
            line_code = " [Line Code]: " + str(line_code)
            if message_orientation == "horizontal":
                print("An error occurred:{};{};{};{};{};{};{}".format(time_stamp, file_name, procedure_name,
                                                                      error_message, error_type, line_number,
                                                                      line_code))
            elif message_orientation == "vertical":
                print("An error occurred:\n{}\n{}\n{}\n{}\n{}\n{}\n{}".format(time_stamp, file_name, procedure_name,
                                                                              error_message, error_type, line_number,
                                                                              line_code))
            else:
                pass
        except Exception:
            pass
