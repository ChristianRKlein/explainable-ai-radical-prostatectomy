from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from shapley_sampling import run_shapley_sampling


# =============================================================================
# --- CONFIGURATION & PATHS ---
# =============================================================================
# Dynamic path resolution: moves up to the main project directory and targets the /data folder
datadir = Path(__file__).resolve().parent.parent / "data"
datafile_input = 'input_data_not_imputed.csv'
datafile_target = 'targets.csv'
target_feature = 'Schnellschnittja_nein'    # Clinical target outcome (Hemoglobin Drop)

# Classifier selection: 'boost', 'rf', 'svm', 'nn' (MLP)
classifier_flag = 'rf'
scale_flag = True      # Enable/disable Z-score standardization
seed = 2056            # Global random state for reproducibility
num_folds = 5          # 5-fold cross-validation setup

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

# Drop blacklisted features with high NaN rates
x = data_no_nan.drop([target_feature, 'Unnamed: 0'], axis=1, errors='ignore')

# =============================================================================
# --- CROSS-VALIDATION SETUP ---
# =============================================================================
# Outer loop setup for unbiased performance evaluation
outer_cv = KFold(n_splits=num_folds, shuffle=True, random_state=42)
count_out = 0
shapley_total = {}


# =============================================================================
# --- NESTED CV TRAINING & EVALUATION LOOP ---
# =============================================================================
for train_index, test_index in outer_cv.split(x,y):
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
        # Re-convert numpy outputs back to dataframes to preserve column headers for Shapley analysis
        X_train_out = pd.DataFrame(scaler.transform(X_train_out), columns=feature_names_encoded)
        X_test_out = pd.DataFrame(scaler.transform(X_test_out), columns=feature_names_encoded)

    # --- CLASSIFIER SELECTION & HYPERPARAMETER GRIDS (INNER CV) ---
    if classifier_flag == 'boost':
        clf = GradientBoostingClassifier(n_estimators=100, random_state=seed)
        param_grid = {
            "learning_rate": [i / 100.0 for i in range(1, 23, 3)],
            "loss": ["log_loss"],
        }
    elif classifier_flag == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=seed)
        param_grid = {
            "criterion": ["gini", "entropy", "log_loss"]
        }
    elif classifier_flag == 'svm':
        clf = SVC(probability=True)
        param_grid = {
            "C": [10 ** i for i in range(-3, 3, 1)],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear", "poly", "sigmoid"],
            "degree": np.arange(2, 3)
        }
    elif classifier_flag == 'nn':
        clf = MLPClassifier(batch_size=32, solver="lbfgs", random_state=seed, max_iter=50000)
        param_grid = {
            "alpha": [1.0, 1.3, 1.6, 2.2, 10.0, 50.0],
            "hidden_layer_sizes": [(10,), (30,), (50,), (20,20)],
            "activation": ['identity','relu','logistic']
        }

    # --- INNER CV GRID SEARCH ---
    # Optimizes hyperparameters using an inner 5-fold cross-validation loop on the training set
    model = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    model.fit(X_train_out, y_train_out)
    print("Best estimator found with parameters:", model.best_params_)
    
    # --- PREDICTION ---
    y_pred = model.predict(X_test_out)

    # =============================================================================
    # --- SHAPLEY SAMPLING COMPUTATION ---
    # =============================================================================
    np.random.seed(14123)
    samplings = []
    runs = 2**13  # Number of permutations for estimation
    def callback(snapshot):
        print ("Current %d" % snapshot[0])
        samplings.append((snapshot[0], snapshot[1]))


    # Compute Shapley feature attributions for the current outer test split
    a_sampling = run_shapley_sampling(model, X_test_out, runs=runs, class_flag= True, callback=callback)
    a_sampling_all = a_sampling.sum(0)  # Aggregate values across all patients within this fold

    k = a_sampling_all.shape[0]
    index_sampling_pairs = [[feature_names_encoded[int(i)], a_sampling_all[i]] for i in range(k)]

    # --- DUMMY ATTRIBUTION AGGREGATION ---
    # Re-bundles One-Hot encoded sub-dummies back into their parent clinical feature
    # (e.g., sums up 'Vortherapien_0' + 'Vortherapien_1' into 'Vortherapien')
    index_sampling_pairs_trimmed = []
    count = 0
    for j in categorical_variables:
        f_len = len(j)
        flag_first = True
        for i in range(k):
            # param could be the same
            if len(index_sampling_pairs[i][0]) > f_len:
                # param is the same
                temp_string = index_sampling_pairs[i][0][0:f_len]
                if temp_string == j:
                    # seen for the first time: append and mark position
                    if flag_first:
                        flag_first = False
                        save_index = count
                        index_sampling_pairs_trimmed.append(index_sampling_pairs[i])
                        index_sampling_pairs_trimmed[save_index][0] = j
                        count += 1
                    # already seen: just add the shapley value
                    else:
                        index_sampling_pairs_trimmed[save_index][1] += index_sampling_pairs[i][1]

    # --- RE-APPEND CONTINUOUS / ORDINAL VARIABLES ---
    remaining_variables = ['Alter', 'BMI',  'HoechsterGleason', 'praeopProstatavolumen', 'iPSA', 'ASA',
                               'ISUPpraeop',  'praeopHaem', 'praeoperativerHB']

    for j in remaining_variables:
        f_len = len(j)
        for i in range(k):
            if len(index_sampling_pairs[i][0]) == f_len:
                # param is the same
                temp_string = index_sampling_pairs[i][0][0:f_len]
                if temp_string == j:
                    index_sampling_pairs_trimmed.append(index_sampling_pairs[i])

    # Rank local feature attributions within the current cross-validation slice
    index_sampling_pairs_trimmed.sort(key=lambda x:x[1])

    # --- GLOBAL ATTRIBUTION COMPILATION ---
    # Accumulate local attribution scores into a global dictionary across all folds
    for key, value in index_sampling_pairs_trimmed:
        val_float = float(value) if isinstance(value, str) else value
        if key in shapley_total:
            shapley_total[key] += val_float
        else:
            shapley_total[key] = val_float

    count_out = count_out + 1

# =============================================================================
# --- EXCEL EXPORT & VISUALIZATION ---
# =============================================================================
# Generate final globally ranked attribution array
final_array = [[key, value] for key, value in shapley_total.items()]
sorted_array = sorted(final_array, key=lambda x: x[1])

# Export results to an Excel spreadsheet for academic archiving
df = pd.DataFrame.from_dict(shapley_total, orient='index', columns=[target_feature])
df = df.sort_values(by=target_feature, ascending=False)
output_excel_filename = f"nested_{target_feature}_{classifier_flag.upper()}_result.xlsx"
df.to_excel(output_excel_filename, index=True, index_label="Keys")
print(f"Global results successfully archived in: {output_excel_filename}")

# --- FEATURE IMPORTANCE PLOT (MATPLOTLIB) ---
feature_arr = [item[0] for item in sorted_array]
value_arr = np.array([item[1] for item in sorted_array])
total_sum = np.sum(value_arr)

# Normalize attribution scores to display relative percentage shares of model impact
if total_sum != 0:
    value_arr = np.flip(value_arr) / total_sum
else:
    value_arr = np.flip(value_arr)
feature_arr.reverse()

fig, axs = plt.subplots(figsize=(9, 5), layout='constrained')
plt.suptitle(f'{target_feature}_test')
plt.ylabel("Normalized Shapley values")
axs.bar(feature_arr, value_arr)
plt.setp(axs.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()

# Print text-based feature ranking list into execution log
top_features_ordered = [item[0] for item in reversed(sorted_array)]
print("\nGlobal Feature Importance Ranking (Highest to Lowest):")
for idx, feat in enumerate(top_features_ordered, 1):
    print(f"{idx}. {feat}")
