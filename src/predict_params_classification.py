from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# =============================================================================
# --- CONFIGURATION & PATHS ---
# =============================================================================
# Dynamic path resolution: moves up to the main project directory and targets the /data folder
datadir = Path(__file__).resolve().parent.parent / "data"
datafile_input = 'input_data_not_imputed.csv'
datafile_target = 'targets.csv'
target_feature = 'Schnellschnittja_nein'

# Classifier selection: 'boost', 'rf', 'svm', 'nn' (MLP), 'mode', 'lr'
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
score_acc = []
score_auc = []
score_sens = []
score_spec = []
score_ppv = []
score_npv = []
all_preds, all_gts = [], []

fold_nr = 1  # Counter for the folds
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
        X_train_out = scaler.transform(X_train_out)
        X_test_out = scaler.transform(X_test_out)

    if classifier_flag == 'lr':
        # Statistical logistic regression as baseline
        model = LogisticRegression(max_iter=1000, random_state=seed)
    else:
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
        elif classifier_flag == 'mode':
            clf = DummyClassifier()
            param_grid = {
                "strategy": ["most_frequent"] 
            }

        # --- INNER CV GRID SEARCH ---
        # Optimizes hyperparameters using an inner 5-fold cross-validation loop on the training set
        model = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    model.fit(X_train_out, y_train_out)
    
    # --- PREDICTION ---
    y_pred = model.predict(X_test_out)
    
    # Calculate probabilities for AUC
    if classifier_flag == 'mode':
        # DummyClassifier (most_frequent) does not output meaningful probabilities for AUC
        y_prob = None
    else:
        y_prob = model.predict_proba(X_test_out)

    # Calculate metrics
    acc = round(100 * metrics.accuracy_score(y_test_out, y_pred),2)###besser
    score_acc.append(acc)

    # AUC calculation (Handling binary vs. multi-class scenarios)
    n_classes = len(np.unique(y_train_out))
    if y_prob is not None:
        try:
            if n_classes == 2:
                # For binary classification, the probability of the positive class (index 1) is sufficient
                # TODO: Verify which class is designated as positive!
                auc_val = metrics.roc_auc_score(y_test_out, y_prob[:, 1])
            else:
                # multiclass='ovr' computes AUC for each class against the rest.
                # average='macro' calculates the unweighted mean of these values.
                auc_val = metrics.roc_auc_score(y_test_out, y_prob, multi_class='ovr', average='macro')
            auc = round(100 * auc_val, 2)
        except Exception as e:
            # Handles edge cases where a fold does not contain all classes or the DummyClassifier fails
            auc = np.nan
            print(f"AUC could not be calculated for this fold. Error: {e}")
    else:
        auc = np.nan
        print("AUC is not defined for DummyClassifier (mode).")
        
    score_auc.append(auc)

    # Calculate sensitivity, specificity, PPV, NPV (Automatic binary/multi-class switch)
    classes = np.unique(y_train_out)
    n_classes = len(classes)
    
    if n_classes == 2:
        # --- PURE BINARY CASE (Focus on the positive class '1' or the higher class) ---
        # Sorting ensures that index 1 corresponds to the positive class
        sorted_classes = np.sort(classes)
        pos_label = sorted_classes[1]
        neg_label = sorted_classes[0]
        
        # Extraction of classic 2x2 confusion matrix elements
        tp = np.sum((y_test_out == pos_label) & (y_pred == pos_label))
        tn = np.sum((y_test_out == neg_label) & (y_pred == neg_label))
        fp = np.sum((y_test_out == neg_label) & (y_pred == pos_label))
        fn = np.sum((y_test_out == pos_label) & (y_pred == neg_label))
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        score_sens.append(round(100 * sens, 2))
        score_spec.append(round(100 * spec, 2))
        score_ppv.append(round(100 * ppv, 2))
        score_npv.append(round(100 * npv, 2))
        
    else:
        # --- MULTI-CLASS CASE (Macro-Averaged OvR as before) ---
        mcm = multilabel_confusion_matrix(y_test_out, y_pred)
        fold_sens, fold_spec, fold_ppv, fold_npv = [], [], [], []
        
        for class_matrix in mcm:
            tn, fp, fn, tp = class_matrix.ravel()
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            fold_sens.append(sens)
            fold_spec.append(spec)
            fold_ppv.append(ppv)
            fold_npv.append(npv)
        
        score_sens.append(round(100 * np.mean(fold_sens), 2))
        score_spec.append(round(100 * np.mean(fold_spec), 2))
        score_ppv.append(round(100 * np.mean(fold_ppv), 2))
        score_npv.append(round(100 * np.mean(fold_npv), 2))

    print(f"Fold Metrics OUTER FOLD {fold_nr}/{num_folds} -> Accuracy: {acc}%, AUC: {f'{auc}%' if not np.isnan(auc) else 'N/A'}")

    all_preds.append(y_pred)
    all_gts.append(y_test_out.to_numpy())
    fold_nr += 1

# =============================================================================
# --- OUTPUT & EXPORT ---
# =============================================================================
print(f"\nResults ({classifier_flag.upper()}):")
print(f"Accuracy: {np.nanmean(score_acc):.1f}% ± {np.nanstd(score_acc):.1f}%")

# Compute Mean and Std for AUC if it contains valid numerical values (not completely NaN)
if not np.all(np.isnan(score_auc)):
    print(f"AUC: {np.nanmean(score_auc):.1f}% ± {np.nanstd(score_auc):.1f}%")
else:
    print("AUC: N/A")

print(f"Sensitivity: {np.mean(score_sens):.1f}% ± {np.nanstd(score_sens):.1f}%  [Recall / TPR]")
print(f"Specificity: {np.mean(score_spec):.1f}% ± {np.nanstd(score_spec):.1f}%  [TNR]")
print(f"PPV (Precision): {np.mean(score_ppv):.1f}% ± {np.nanstd(score_ppv):.1f}%")
print(f"NPV: {np.mean(score_npv):.1f}% ± {np.nanstd(score_npv):.1f}%")
