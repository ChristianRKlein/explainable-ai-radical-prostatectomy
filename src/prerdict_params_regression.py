from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# =============================================================================
# --- METRIC DEFINITIONS ---
# =============================================================================
def symmetric_mean_absolute_percentage_error_100(y_true, y_pred):
    """
    Computes the 0-100% version of SMAPE.
    Returns a value between 0 and 100 (%).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Denominator WITHOUT the division by 2
    denominator = np.abs(y_true) + np.abs(y_pred)
    
    # Handle division by zero (when both true and pred are exactly 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        smape_elements = np.where(denominator != 0, np.abs(y_pred - y_true) / denominator, 0.0)
    
    # Compute the mean and scale to percentage
    return np.mean(smape_elements) * 100

# =============================================================================
# --- CONFIGURATION & PATHS ---
# =============================================================================
# Dynamic path resolution: moves up to the main project directory and targets the /data folder
datadir = Path(__file__).resolve().parent.parent / "data" 
datafile_input = 'input_data_not_imputed.csv'
datafile_target = 'targets.csv'
target_feature = 'postopHB' # Clinical target outcome (Hemoglobin Drop)

# Regressor selection: 'boost', 'rf', 'svm', 'nn' (MLP)
regressor_flag = 'rf'  
scale_flag = True      # Enable/disable Z-score standardization
seed = 2056            # Global random state for reproducibility
num_folds = 5          # 5-fold cross-validation setup
r = 5                  # Rounding decimals for ordinal targets (r = 0)

# Cohort routing: 0 = Combined data, 1 = RARP cohort only, 2 = ORP cohort only
rarp = 0

# Clinical data type definitions for correct operational pipeline handling
categorical_variables = ['RARP', 'DRUsuspekt', 'PraeopcTx','DAmico', 'Vortherapien', 'Vortherapien_welche', 'Präoperative_ADTBehandlung', 'Nerverhalt', 'Nerverhaltwo']
ordinal_variables = ['ISUPpraeop','ASA']
# Feature blacklist: clinical variables excluded due to excessive missing values (NaNs)
prob_to_drop_variables = ['ICIQ', 'ICIQ1Kontinenz', 'ICIQ2Lebensqualitaet', 'ICIQ3Gesamtzustand', 'IPSS', 'IIEF', 'Hornheider', 'Nerverhaltwo_vereinfacht']     #, 'Nerverhalt', 'Nerverhaltwo'

# =============================================================================
# --- DATA LOADING & FILTERING ---
# =============================================================================
data_input = pd.read_csv(pjoin(datadir, datafile_input), sep=',')
data_target = pd.read_csv(pjoin(datadir, datafile_target), sep=';', encoding="ISO-8859-1")

y = data_target[target_feature]
x = data_input.copy()

# Cohort stratification based on surgical approach (RARP vs. ORP)
if rarp == 1 or rarp == 2:
    # 1. Identify row indices matching the targeted surgical cohort
    rarp_1_indices = x[x['RARP'] == rarp].index

    # 2. Filter both features (x) and target (y) strictly on these indices
    x = x.loc[rarp_1_indices].reset_index(drop=True)
    y = y.loc[rarp_1_indices].reset_index(drop=True)

    # 3. Drop 'RARP' column as its variance is now 0 within this cohort
    if 'RARP' in x.columns:
        x = x.drop(columns=['RARP'])
        # Remove from configuration list to avoid errors down the pipeline
        if 'RARP' in categorical_variables:
            categorical_variables.remove('RARP')

    print(f"Remaining patients in filtered cohort (RARP={rarp}): {len(x)}")

# =============================================================================
# --- DATA PREPROCESSING (GLOBAL) ---
# =============================================================================
# Cast categorical columns explicitly to the pandas 'category' dtype for correct One-Hot handling
for variable in categorical_variables:
    if variable in x.columns:
        x[variable] = x[variable].astype("category")

# Listwise deletion of rows containing missing target values (critical for model training)
dataconcat = pd.concat([x, y], axis=1)
data_no_nan = dataconcat.dropna(subset=[target_feature])
y = data_no_nan[target_feature]
x = data_no_nan.drop([target_feature, 'Unnamed: 0'], axis=1, errors='ignore')

# Drop blacklisted features with high NaN rates
x = x.drop(columns=[v for v in prob_to_drop_variables if v in x.columns])

# =============================================================================
# --- CROSS-VALIDATION SETUP ---
# =============================================================================
# Outer loop setup for unbiased performance evaluation
outer_cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)
score_mae, score_rmse, score_r2, score_mape, score_smape = [], [], [], [], []
all_preds, all_gts = [], []

# =============================================================================
# --- NESTED CV TRAINING & EVALUATION LOOP ---
# =============================================================================
for train_index, test_index in outer_cv.split(x):
    X_train_out, X_test_out = x.iloc[train_index].copy(), x.iloc[test_index].copy()
    y_train_out, y_test_out = y.iloc[train_index], y.iloc[test_index]

    # --- IN-LOOP IMPUTATION (Strictly prevents Data Leakage) ---
    # Imputation parameters are computed ONLY on X_train_out and mapped onto X_test_out
    for col in X_train_out.columns:
        if col in categorical_variables:
            # Impute categorical variables using the training fold's mode
            fill_value = X_train_out[col].mode()[0]
            X_train_out[col] = X_train_out[col].fillna(fill_value)
            X_test_out[col] = X_test_out[col].fillna(fill_value)
        else:
            # Impute continuous and ordinal variables using the training fold's mean
            fill_value = X_train_out[col].mean()
            X_train_out[col] = X_train_out[col].fillna(fill_value)
            X_test_out[col] = X_test_out[col].fillna(fill_value)

    # Round ordinal values post-imputation back to integer scales
    X_train_out[ordinal_variables] = X_train_out[ordinal_variables].round()
    X_test_out[ordinal_variables] = X_test_out[ordinal_variables].round()

    # --- IN-LOOP ONE-HOT ENCODING ---
    # Done inside the loop to avoid giving the training matrix context about unknown test categories
    combined = pd.concat([X_train_out, X_test_out], axis=0)
    cats_to_encode = [v for v in categorical_variables if v in combined.columns]
    combined = pd.get_dummies(combined, columns=cats_to_encode)
    
    feature_names_encoded = list(combined.columns)
    X_train_out = combined.iloc[:len(X_train_out)]
    X_test_out = combined.iloc[len(X_train_out):]

    # --- IN-LOOP Z-SCORE STANDARDIZATION ---
    if scale_flag:
        scaler = StandardScaler()
        scaler.fit(X_train_out) # Fit strictly limited to the training split
        X_train_out = scaler.transform(X_train_out)
        X_test_out = scaler.transform(X_test_out)

    if regressor_flag == 'lr':
            # Statistical linear model as baseline
            model = Ridge(alpha=1.0, max_iter=10000, random_state=seed)
    else:
        # --- REGRESSOR SELECTION & HYPERPARAMETER GRIDS (INNER CV) ---
        if regressor_flag == 'boost':
            reg = GradientBoostingRegressor(n_estimators=100, random_state=seed)
            param_grid = {
                "learning_rate": [i / 100.0 for i in range(1, 23, 3)],
                "loss": ["squared_error", "absolute_error", "huber"],
            }
        elif regressor_flag == 'rf':
            reg = RandomForestRegressor(n_estimators=100, random_state=seed)
            param_grid = {
                "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"]
            }
        elif regressor_flag == 'svm':
            reg = SVR()
            param_grid = {
                "C": [10 ** i for i in range(-3, 3, 1)],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf", "linear", "poly", "sigmoid"],
                "degree": np.arange(2, 5)
            }
        elif regressor_flag == 'nn':
            # Multilayer Perceptron utilizing lbfgs solver tailored for small clinical cohorts
            reg = MLPRegressor(batch_size=32, solver="lbfgs", random_state=seed, max_iter=50000) 
            param_grid = {
                "alpha": [1.0, 1.3, 1.6, 2.2, 10.0, 50.0],
                "hidden_layer_sizes": [(10,), (30,), (50,), (20,20)],
                "activation": ['identity','tanh','relu']
            }
        elif regressor_flag == 'mean':
            reg = DummyRegressor(strategy='mean')
            param_grid = {
                "strategy": ["mean"]
            }
        elif regressor_flag == 'median':
            reg = DummyRegressor(strategy='median')
            param_grid = {
                "strategy": ["median"]
            }

        
        # --- INNER CV GRID SEARCH ---
        # Optimizes hyperparameters using an inner 5-fold cross-validation loop on the training set
        model = GridSearchCV(reg, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    model.fit(X_train_out, y_train_out)
    
    # --- PREDICTION ---
    y_pred = model.predict(X_test_out).round(r)

    # Calculate validation metrics
    errors = np.abs(y_pred - y_test_out)
    rmse = np.sqrt(mean_squared_error(y_test_out, y_pred))
    mape_acc = 100 - (100 * np.mean(errors / y_test_out))
    r2 = max(0, r2_score(y_test_out, y_pred) * 100)
    smape_acc = 100 - symmetric_mean_absolute_percentage_error_100(y_test_out, y_pred)

    score_mae.append(np.mean(errors))
    score_rmse.append(rmse)
    score_mape.append(mape_acc)
    score_r2.append(r2)
    score_smape.append(smape_acc)
    
    all_preds.append(y_pred)
    all_gts.append(y_test_out.to_numpy())

# =============================================================================
# --- OUTPUT & EXPORT ---
# =============================================================================
print(f"\nResults ({regressor_flag.upper()}):")
print(f"MAE:            {np.mean(score_mae):.2f} ± {np.std(score_mae):.2f}")
print(f"RMSE:           {np.mean(score_rmse):.2f} ± {np.std(score_rmse):.2f}")
print(f"Accuracy (MAPE): {np.mean(score_mape):.1f}% ± {np.std(score_mape):.1f}%")
print(f"SMAPE:          {np.mean(score_smape):.1f}% ± {np.std(score_smape):.1f}%")
print(f"R2:             {np.mean(score_r2):.1f}% ± {np.std(score_r2):.1f}%")
