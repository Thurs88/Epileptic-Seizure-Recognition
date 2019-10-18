import gc

gc.enable()

import config
from classifier import EEGClassifier

from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def main():
    eeg_class = EEGClassifier()
    # load data
    df = eeg_class.read_data(csv_file_name=config.FILE_NAME,
                             csv_file_path=config.FILE_PATH,
                             reduce_memory=config.REDUCE_MEMORY)
    # show data info
    eeg_class.show_file_information(df)
    eeg_class.show_file_data(df)
    eeg_class.show_descriptive_statistics(df)
    eeg_class.check_missing_values(df)

    # data preparation
    df_prepared = eeg_class.data_preparation(df)
    # feature engineering
    df_nf = eeg_class.feature_extract(df_prepared)
    X, y = eeg_class.select_label_target(df_nf, config.TARGET_COLUMN_NAME)
    # split data to train/test
    X_train, X_test, y_train, y_test = eeg_class.split_data(X,
                                                            y,
                                                            test_size_percentage=config.TEST_SIZE_PERC,
                                                            random_state=config.RANDOM_STATE)
    # create cross-val groups
    gkf = eeg_class.cv_folds(X_train, y_train, splits=config.N_SPLITS)
    # SVM classificator
    clf = eeg_class.svm_model(
        kernel=config.KERNEL,
        gamma=config.GAMMA,
        C=config.C,
        class_weight=config.CLASS_WEIGHT
    )
    # create pipeline
    pipe = eeg_class.create_pipeline(
        eeg_class.data_scaler(scaler_type=config.DATA_STANDARD_SCALER),
        eeg_class.reduce_dimension(n=config.PCs),
        clf
    )
    # get cv-score
    cv_score = eeg_class.cv_score(pipe,
                                  X_train, y_train,
                                  scoring=config.SCORING,
                                  folds=gkf,
                                  n_jobs=config.N_JOBS
                                  )
    # model prediction
    y_prediction = eeg_class.test_model(pipe, X_test)
    # save predictions
    eeg_class.get_submission(config.RESULT_PATH, config.RESULT_FILE_NAME, y_prediction)
    # model evaluation
    eeg_class.evaluate_model(y_test, y_prediction)


if __name__ == '__main__':
    main()
