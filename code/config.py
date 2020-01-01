# Input
FILE_PATH = "../data/"
FILE_NAME = "data.csv"
# Output
RESULT_PATH = "../results/"
RESULT_FILE_NAME = "eeg_predictions.csv"

TARGET_COLUMN_NAME = "y"
REDUCE_MEMORY = True
PCs = 50  # n-components for PCA
N_SPLITS = 10  # n-folds for cross-val
TEST_SIZE_PERC = 20  # test size in percentage (%)
RANDOM_STATE = 1

# Types of Scalers
DATA_STANDARD_SCALER="StandardScaler"
DATA_ROBUST_SCALER ="RobustScaler"
DATA_NORMALIZER_SCALER ="Normalizer"
DATA_MIN_MAX_SCALER="MinMaxScaler"
DATA_MAX_ABS_SCALER="MaxAbsScaler"

# Classificator params
SCORING = 'roc_auc'
KERNEL = 'rbf'
GAMMA = 'auto'
C = 1
CLASS_WEIGHT = 'balanced'
N_JOBS = -1
