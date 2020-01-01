import config
from datetime import datetime
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from preprocessing import DataPreprocessor


class EEGClassifier(DataPreprocessor):
    """
    Класс для сборки пайплайна обработки данных,
    выбора модели и оценки результатов
    """

    def __init__(self):
        super().__init__()

    def timeit(func):
        def wrapper(*args, **kwargs):
            start = datetime.now()
            res = func(*args, **kwargs)
            print(f'{func.__name__} done! Time:', datetime.now() - start)
            return res
        return wrapper

    def svm_model(self, kernel, gamma, C, class_weight):
        """
        initial the SVM classifier for pipeline
        :param kernel: specifies the kernel type to be used in the algorithm.
        :param gamma: kernel coefficient
        :param C: regularization parameter
        :param class_weight: class weights in case of unbalanced data
        :return SVM object
        """
        try:
            svm_classifier = SVC(kernel=kernel, gamma=gamma,
                                 C=C, class_weight=class_weight)
        except Exception:
            self.print_exception_message()
        return svm_classifier

    def create_pipeline(self, scaler, reduce_dimension, classifier):
        """
        create pipeline for classification
        :param scaler: data_scaler func
        :param reduce_dimension: reduce_dimension func
        :param classifier: svm_model func
        :return pipeline object
        """
        print('create pipeline...')
        pipe = Pipeline(
            [('scale', scaler),
             ('pca', reduce_dimension),
             ('clf', classifier)]
        )
        return pipe

    @timeit
    def cv_score(self, pipe, X_train, y_train, scoring, folds, n_jobs):
        """
        evaluate model
        :param pipe: model pipeline
        :param X_train: X train scaled
        :param y_train: train target values
        :param scoring: evaluation metric
        :param folds: number of cv-folds
        :param n_jobs: number of jobs to run in parallel
        :return none
        """
        print('evaluate model...')
        try:
            y_pred = cross_val_score(pipe, X_train, y_train, cv=folds, scoring=scoring, n_jobs=n_jobs)
            pipe.fit(X_train, y_train)
            print('--------------CV RESULTS--------------')
            print()
            print(f'Mean {scoring}: ', np.mean(y_pred))
            print(f'{scoring} std: ', np.std(y_pred))
            print('--------------------------------------')
            print()
        except Exception:
            self.print_exception_message()
        return pipe

    def test_model(self, pipe, X_test):
        """
        predict target on test data
        :param pipe: model pipeline
        :param X_test: test dataframe
        :return predicted y
        """
        print('test model...')
        try:
            y_pred = pipe.predict(X_test)
        except Exception:
            self.print_exception_message()
        return y_pred

    def evaluate_model(self, y_test, y_pred):
        """
        print confusion matrix, classification report and accuracy score values
        :param y_test: test target values
        :param y_pred: predicted y
        :return none
        """
        print('--------------TEST RESULTS--------------')
        print()
        try:
            confusion_matrix_value = confusion_matrix(y_test, y_pred)
            print("CONFUSION MATRIX")
            print(confusion_matrix_value)
            print()

            classification_report_result = classification_report(y_test, y_pred)
            print('CLASSIFICATION REPORT')
            print(classification_report_result)
            print()

            roc_score_value = roc_auc_score(y_test, y_pred) * 100
            roc_score_value = float("{0:.2f}".format(roc_score_value))
            print("ROC SCORE")
            print("{} %".format(roc_score_value))
            print()

            accuracy_score_value = accuracy_score(y_test, y_pred) * 100
            accuracy_score_value = float("{0:.2f}".format(accuracy_score_value))
            print("ACCURACY SCORE")
            print("{} %".format(accuracy_score_value))
            print('---------------------------------------')
            print()
        except Exception:
            self.print_exception_message()
