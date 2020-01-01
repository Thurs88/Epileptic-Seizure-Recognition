import sys
import time
import traceback
import numpy as np
import pandas as pd


class rootClass(object):
    """
    Класс для базовых функций
    """

    def __init__(self):
        pass

    def read_data(self, csv_file_name, csv_file_path, separator=",", reduce_memory=False, encoding_unicode=None):
        """
        read data from a csv file
        :param csv_file_name: csv file name
        :param csv_file_path: csv file path
        :param separator: column separator
        :param reduce_memory: use/not reduce_memory function
        :param encoding_unicode: csv file encoding unicode
        :return dataframe
        """
        df = None
        try:
            if reduce_memory:
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
