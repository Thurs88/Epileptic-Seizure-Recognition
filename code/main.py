import gc
gc.enable()

import config
from classifier import EEGClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def main():
    eeg = EEGClassifier()
    # load data
    df = eeg.read_data(csv_file_name=config.FILE_NAME,
                       csv_file_path=config.FILE_PATH,
                       reduce_memory=config.REDUCE_MEMORY)
    # show data info
    eeg.show_file_information(df)
    eeg.show_file_data(df)
    eeg.show_descriptive_statistics(df)
    eeg.check_missing_values(df)

    # data preparation
    df_prepared = eeg.data_preparation(df)
    # feature engineering
    df_nf = eeg.feature_extract(df_prepared)
    X, y = eeg.select_label_target(df_nf, config.TARGET_COLUMN_NAME)
    # split data to train/test
    X_train, X_test, y_train, y_test = eeg.split_data(X,
                                                      y,
                                                      test_size_percentage=config.TEST_SIZE_PERC,
                                                      random_state=config.RANDOM_STATE)
    # create cross-val groups
    gkf = eeg.cv_folds(X_train, y_train, splits=config.N_SPLITS)
    # SVM classifier
    clf = eeg.svm_model(
        kernel=config.KERNEL,
        gamma=config.GAMMA,
        C=config.C,
        class_weight=config.CLASS_WEIGHT
    )
    # create pipeline
    pipe = eeg.create_pipeline(
        eeg.scaler(scaler_type=config.DATA_STANDARD_SCALER),
        eeg.reduce_dimension(n=config.PCs),
        clf
    )
    # get cv-score
    cv_score = eeg.cv_score(pipe,
                            X_train, y_train,
                            scoring=config.SCORING,
                            folds=gkf,
                            n_jobs=config.N_JOBS
                            )
    # model prediction
    y_prediction = eeg.test_model(pipe, X_test)
    # save predictions
    eeg.get_submission(config.RESULT_PATH, config.RESULT_FILE_NAME, y_prediction)
    # model evaluation
    eeg.evaluate_model(y_test, y_prediction)


if __name__ == '__main__':
    main()
