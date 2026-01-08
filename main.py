
"""
S-N Fatigue Analysis for Offshore Wind Foundation Design
PCA Common Slope Method (from fatigue_analysis_a.py)
PCA_paper Method (from main_16.py)
Spearman and HGVS Feature Selection Methods
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===================================================================
# DATA LOADING AND PREPROCESSING
# ===================================================================

print("="*70)
print("S-N FATIGUE ANALYSIS - OFFSHORE WIND FOUNDATION DESIGN")
print("="*70)

# Load data
df = pd.read_excel('./Dataset_S355.xlsx')

# Delete rows where Runout = 1
df = df[df['Runout'] != 1].copy()

# Delete rows where Batch = 21 or 22
df = df[~df['Batch'].isin([21, 22])].copy()

# Reset index
df.reset_index(drop=True, inplace=True)

# Create log-transformed features
df['logL'] = np.log10(df['Length'])
df['logW'] = np.log10(df['Width'])
df['logT'] = np.log10(df['Thickness'])
df['logSigma'] = np.log10(df['Nom_Srange'])
df['logN'] = np.log10(df['Nf'])

# Define features and target
feature_names = ['logL', 'logW', 'logT', 'logSigma', 'R']
X = df[feature_names].values
y = df['logN'].values
batches = df['Batch'].values

print(f"\nData Shape: {X.shape}")
print(f"Target Shape: {y.shape}")
print(f"\nTarget (logN) Distribution:")
print(f"  Mean: {y.mean():.4f}")
print(f"  Std: {y.std():.4f}")
print(f"  Min: {y.min():.4f}")
print(f"  Max: {y.max():.4f}")
print(f"\nNumber of unique batches: {len(np.unique(batches))}")
print(f"Unique batches: {sorted(np.unique(batches))}")

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate RSE, R2, and MAE"""
    rse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rse, r2, mae

def spearman_feature_selection(X, y, feature_names, top_k=4):
    """Select top k features based on Spearman correlation"""
    correlations = []
    for i in range(X.shape[1]):
        corr, _ = spearmanr(X[:, i], y)
        correlations.append(abs(corr))

    top_indices = np.argsort(correlations)[::-1][:top_k]
    return top_indices, [feature_names[i] for i in top_indices]

def hgvs_feature_selection(X, y, feature_names, top_k=4):
    """HGVS (High Gradient Value Selection)"""
    # Using correlation-based importance similar to HVGS in main_16.py
    importances = []
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        importances.append(abs(corr))

    top_indices = np.argsort(importances)[::-1][:top_k]
    return top_indices, [feature_names[i] for i in top_indices]

def get_feature_importance(model, feature_names, model_type):
    """Get feature ranking based on model type"""
    if model_type == 'linear':
        importances = np.abs(model.coef_)
    else:  # random forest
        importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]
    return [feature_names[i] for i in indices]

# ===================================================================
# PCA_PAPER METHOD
# ===================================================================

def pca_paper_loocv(X, y, batches, feature_names, model_type='linear', exclude_feature=None):
    """
    PCA Paper method with Common Slope - transform all data including logN in one batch
    Modified to use Common Slope approach from main_16.py
    """
    if exclude_feature is not None:
        exclude_idx = feature_names.index(exclude_feature)
        feature_mask = [i for i in range(len(feature_names)) if i != exclude_idx]
        X_used = X[:, feature_mask]
        features_used = [feature_names[i] for i in feature_mask]
    else:
        X_used = X.copy()
        features_used = feature_names.copy()

    # PCA on all data including target (key difference from PCA Common)
    features_with_target = np.column_stack([X_used, y])
    # features_with_target = np.column_stack([X_used, y])

    # Standardize all data at once
    scaler_global = StandardScaler()
    X_all_scaled = scaler_global.fit_transform(features_with_target)

    # PCA on all data
    # pca_global = PCA(n_components=min(6, features_with_target.shape[1]))
    pca_global = PCA()
    pca_global.fit(X_all_scaled)

    pc_explained_var = pca_global.explained_variance_ratio_

    unique_batches = np.unique(batches)
    predictions = []
    actuals = []

    # Leave-One-Batch-Out Cross-Validation
    for test_batch in unique_batches:
        test_mask = (batches == test_batch)
        train_mask = ~test_mask

        X_train_orig = X_used[train_mask]
        X_test_orig = X_used[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        batch_train = batches[train_mask]

        if len(X_train_orig) < 5 or len(X_test_orig) == 0:
            continue

        # Combine features with target for PCA
        X_train_with_target = np.column_stack([X_train_orig, y_train])
        X_test_with_target = np.column_stack([X_test_orig, y_test])

        # Apply global scaler and PCA
        X_train_scaled = scaler_global.transform(X_train_with_target)
        X_test_scaled = scaler_global.transform(X_test_with_target)

        X_train_pca = pca_global.transform(X_train_scaled)
        X_test_pca = pca_global.transform(X_test_scaled)

        # Keep only PC1 and PC2
        X_train_filt_pca = X_train_pca.copy()
        X_train_filt_pca[:, 2:] = 0
        X_test_filt_pca = X_test_pca.copy()
        X_test_filt_pca[:, 2:] = 0

        # Inverse transform
        X_train_filt_scaled = pca_global.inverse_transform(X_train_filt_pca)
        X_test_filt_scaled = pca_global.inverse_transform(X_test_filt_pca)

        X_train_filt = scaler_global.inverse_transform(X_train_filt_scaled)
        X_test_filt = scaler_global.inverse_transform(X_test_filt_scaled)

        # Extract features only (excluding target column)
        X_train_features = X_train_filt[:, :-1]
        X_test_features = X_test_filt[:, :-1]

        # === Common Slope Method (from main_16.py) ===
        # Find logSigma index in the features
        if 'logSigma' in features_used:
            logSigma_idx = features_used.index('logSigma')
        else:
            # If logSigma not in features, use standard regression
            if model_type == 'linear':
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_features, y_train)
            y_pred = model.predict(X_test_features)
            predictions.extend(y_pred)
            actuals.extend(y_test)
            continue

        # Get logN from the filtered data (last column)
        y_train_filt = X_train_filt[:, -1]

        # Calculate common slope from training batches
        train_unique_batches = np.unique(batch_train)
        batch_slopes = []

        for batch_id in train_unique_batches:
            batch_mask = (batch_train == batch_id)
            if np.sum(batch_mask) >= 2:
                X_batch = X_train_filt[batch_mask]
                lr_b = LinearRegression()
                lr_b.fit(X_batch[:, logSigma_idx].reshape(-1, 1), X_batch[:, -1])  # logN is last column
                batch_slopes.append(lr_b.coef_[0])

        # Average slope across batches
        if len(batch_slopes) > 0:
            alpha_common = np.mean(batch_slopes)
        else:
            alpha_common = -2.326  # Default value if no batches available

        # Use one point for intercept adjustment (last training point)
        # point_i_logSigma_filt = X_train_filt[-1, logSigma_idx]
        # point_i_logN = y_train_filt[-1]

        # 改善案: 高応力域（低サイクル疲労）の点を選択
        high_stress_mask = X_train_filt[:, logSigma_idx] > np.median(X_train_filt[:, logSigma_idx])
        if np.sum(high_stress_mask) > 0:
            high_stress_indices = np.where(high_stress_mask)[0]
            # 最も高いlogSigmaの点を選択
            best_idx = high_stress_indices[np.argmax(X_train_filt[high_stress_indices, logSigma_idx])]
            point_i_logSigma_filt = X_train_filt[best_idx, logSigma_idx]
            point_i_logN = y_train_filt[best_idx]


        # Adjusted intercept
        beta_adj = point_i_logN - alpha_common * point_i_logSigma_filt

        # Prediction using common slope
        y_pred = alpha_common * X_test_filt[:, logSigma_idx] + beta_adj

        predictions.extend(y_pred)
        actuals.extend(y_test)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    rse, r2, mae = calculate_metrics(actuals, predictions)

    # Feature ranking (train on full data)
    X_all_with_target = np.column_stack([X_used, y])
    X_all_scaled = scaler_global.transform(X_all_with_target)
    X_all_pca = pca_global.transform(X_all_scaled)
    X_all_filt_pca = X_all_pca.copy()
    X_all_filt_pca[:, 2:] = 0
    X_all_filt_scaled = pca_global.inverse_transform(X_all_filt_pca)
    X_all_filt = scaler_global.inverse_transform(X_all_filt_scaled)
    X_all_features = X_all_filt[:, :-1]

    if model_type == 'linear':
        model_full = LinearRegression()
    else:
        model_full = RandomForestRegressor(n_estimators=100, random_state=42)
    model_full.fit(X_all_features, y)

    ranking = get_feature_importance(model_full, features_used, model_type)

    return rse, r2, mae, ranking, pc_explained_var


def feature_selection_loocv(X, y, batches, feature_names, method='spearman', model_type='linear', exclude_feature=None):
    """Feature selection methods (Spearman/HGVS) with LOOCV"""
    unique_batches = np.unique(batches)
    predictions = []
    actuals = []
    
    # フォールドごとの特徴選択結果を記録
    feature_counts = {fname: 0 for fname in feature_names}

    # Leave-One-Batch-Out Cross-Validation
    for test_batch in unique_batches:
        test_mask = batches == test_batch
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(X_train) < 10 or len(X_test) == 0:
            continue

        # Select top 4 features
        if method == 'spearman':
            top_indices, top_features = spearman_feature_selection(X_train, y_train, feature_names, top_k=4)
        else:  # HGVS
            top_indices, top_features = hgvs_feature_selection(X_train, y_train, feature_names, top_k=4)

        # 選ばれた特徴をカウント
        for feat in top_features:
            feature_counts[feat] += 1

        # Exclude feature if specified
        if exclude_feature is not None and exclude_feature in top_features:
            top_features_used = [f for f in top_features if f != exclude_feature]
            top_indices = [feature_names.index(f) for f in top_features_used]
        else:
            top_features_used = top_features
            top_indices = [feature_names.index(f) for f in top_features_used]

        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]

        # Train model
        if model_type == 'linear':
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        predictions.extend(y_pred)
        actuals.extend(y_test)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    rse, r2, mae = calculate_metrics(actuals, predictions)

    # === 修正：安定性ベースのランキング ===
    # 各フォールドで選ばれた回数でソート（Consensus Feature Selection）
    ranking = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    ranking = [feat for feat, count in ranking if count > 0]
    
    # exclude_featureを除外
    if exclude_feature is not None and exclude_feature in ranking:
        ranking = [f for f in ranking if f != exclude_feature]

    return rse, r2, mae, ranking


# ===================================================================
# MAIN EXECUTION
# ===================================================================

# ==================== LOOCV WITH LINEAR REGRESSION ====================
print("\n" + "="*70)
print("RUNNING LOOCV WITH LINEAR REGRESSION")
print("="*70)

results_loocv_linear = []

# PCA_paper
print("\nProcessing PCA_paper (Linear LOOCV)...")
# full_rse, full_r2, full_mae, full_ranking, pc_var = pca_common_loocv_with_bootstrap(X, y, batches, feature_names)

full_rse, full_r2, full_mae, full_ranking, pc_var = pca_paper_loocv(X, y, batches, feature_names, model_type='linear')
print(f"  PC1 variance: {pc_var[0]:.4f}, PC2 variance: {pc_var[1]:.4f}")
drop1_rse, drop1_r2, drop1_mae, drop1_ranking, _ = pca_paper_loocv(X, y, batches, feature_names, model_type='linear', exclude_feature=full_ranking[0])
results_loocv_linear.append({
    'Method': 'PCA_paper',
    'Full_RSE': full_rse,
    'DropTop1_RSE': drop1_rse,
    'Full_R2': full_r2,
    'DropTop1_R2': drop1_r2,
    'Full_MAE': full_mae,
    'DropTop1_MAE': drop1_mae,
    'Full_Ranking': ', '.join(full_ranking),
    'DropTop1_Ranking': ', '.join(drop1_ranking)
})



# Spearman
print("\nProcessing Spearman (Linear LOOCV)...")
full_rse, full_r2, full_mae, full_ranking = feature_selection_loocv(X, y, batches, feature_names, method='spearman', model_type='linear')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = feature_selection_loocv(X, y, batches, feature_names, method='spearman', model_type='linear', exclude_feature=full_ranking[0])
results_loocv_linear.append({
    'Method': 'Spearman',
    'Full_RSE': full_rse,
    'DropTop1_RSE': drop1_rse,
    'Full_R2': full_r2,
    'DropTop1_R2': drop1_r2,
    'Full_MAE': full_mae,
    'DropTop1_MAE': drop1_mae,
    'Full_Ranking': ', '.join(full_ranking),
    'DropTop1_Ranking': ', '.join(drop1_ranking)
})

# HGVS
print("\nProcessing HGVS (Linear LOOCV)...")
full_rse, full_r2, full_mae, full_ranking = feature_selection_loocv(X, y, batches, feature_names, method='hgvs', model_type='linear')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = feature_selection_loocv(X, y, batches, feature_names, method='hgvs', model_type='linear', exclude_feature=full_ranking[0])
results_loocv_linear.append({
    'Method': 'HGVS',
    'Full_RSE': full_rse,
    'DropTop1_RSE': drop1_rse,
    'Full_R2': full_r2,
    'DropTop1_R2': drop1_r2,
    'Full_MAE': full_mae,
    'DropTop1_MAE': drop1_mae,
    'Full_Ranking': ', '.join(full_ranking),
    'DropTop1_Ranking': ', '.join(drop1_ranking)
})

df_loocv_linear = pd.DataFrame(results_loocv_linear)
df_loocv_linear.to_csv('./LOOCV_Linear.csv', index=False)
print("Saved csv")