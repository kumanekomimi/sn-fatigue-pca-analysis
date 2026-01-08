
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
df = pd.read_excel('Dataset_S355.xlsx')

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
# PCA COMMON SLOPE METHOD (from fatigue_analysis_a.py)
# ===================================================================

# def pca_common_loocv(X, y, batches, feature_names, model_type='linear', exclude_feature=None):
    """
    PCA Common Slope with Leave-One-Batch-Out CV
    Reference: fatigue_analysis_a.py
    """
    if exclude_feature is not None:
        exclude_idx = feature_names.index(exclude_feature)
        feature_mask = [i for i in range(len(feature_names)) if i != exclude_idx]
        X_used = X[:, feature_mask]
        features_used = [feature_names[i] for i in feature_mask]
    else:
        X_used = X.copy()
        features_used = feature_names.copy()

    unique_batches = np.unique(batches)
    predictions = []
    actuals = []
    pc_explained_var = None

    # Leave-One-Batch-Out Cross-Validation
    for test_batch in unique_batches:
        test_mask = batches == test_batch
        train_mask = ~test_mask

        X_train, X_test = X_used[train_mask], X_used[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        batch_train = batches[train_mask]

        if len(X_train) < 5 or len(X_test) == 0:
            continue

        # Shuffle training data
        # shuffle_idx = np.random.permutation(len(X_train))
        # X_train = X_train[shuffle_idx]
        # y_train = y_train[shuffle_idx]
        # batch_train = batch_train[shuffle_idx]


        # Standardize (fit on train only)
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        # PCA (fit on train only)
        pca = PCA()
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)

        # Store explained variance (first fold only)
        if pc_explained_var is None:
            pc_explained_var = pca.explained_variance_ratio_

        # Keep only PC1 and PC2
        X_train_pca_reduced = X_train_pca.copy()
        X_test_pca_reduced = X_test_pca.copy()
        X_train_pca_reduced[:, 2:] = 0
        X_test_pca_reduced[:, 2:] = 0

        # Inverse transform back to original space
        X_train_reconstructed = pca.inverse_transform(X_train_pca_reduced)
        X_test_reconstructed = pca.inverse_transform(X_test_pca_reduced)

        # Inverse standardization
        X_train_inv = scaler.inverse_transform(X_train_reconstructed)
        X_test_inv = scaler.inverse_transform(X_test_reconstructed)

        # Calculate common slope from training batches
        train_unique_batches = np.unique(batch_train)
        batch_slopes = []

        for batch_id in train_unique_batches:
            batch_mask = batch_train == batch_id
            if np.sum(batch_mask) > 1:
                X_batch = X_train_inv[batch_mask]
                y_batch = y_train[batch_mask]

                model_batch = LinearRegression()
                model_batch.fit(X_batch, y_batch)
                batch_slopes.append(model_batch.coef_)

        # Average slope across batches
        if len(batch_slopes) > 0:
            common_slope = np.mean(batch_slopes, axis=0)
        else:
            common_slope = None

        # Train final model
        if model_type == 'linear':
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        model.fit(X_train_inv, y_train)
        y_pred = model.predict(X_test_inv)

        predictions.extend(y_pred)
        actuals.extend(y_test)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    rse, r2, mae = calculate_metrics(actuals, predictions)

    # Feature ranking (train on full data)
    scaler_full = StandardScaler()
    X_full_std = scaler_full.fit_transform(X_used)
    pca_full = PCA()
    X_full_pca = pca_full.fit_transform(X_full_std)
    X_full_pca_reduced = X_full_pca.copy()
    X_full_pca_reduced[:, 2:] = 0
    X_full_reconstructed = pca_full.inverse_transform(X_full_pca_reduced)
    X_full_inv = scaler_full.inverse_transform(X_full_reconstructed)

    if model_type == 'linear':
        model_full = LinearRegression()
    else:
        model_full = RandomForestRegressor(n_estimators=100, random_state=42)

    model_full.fit(X_full_inv, y)
    ranking = get_feature_importance(model_full, features_used, model_type)

    return rse, r2, mae, ranking, pc_explained_var
def pca_common_loocv(X, y, batches, feature_names, model_type='linear', exclude_feature=None):
    """
    PCA Common Slope with Leave-One-Batch-Out CV
    論文のCommon Slope手法を実装
    """
    if exclude_feature is not None:
        exclude_idx = feature_names.index(exclude_feature)
        feature_mask = [i for i in range(len(feature_names)) if i != exclude_idx]
        X_used = X[:, feature_mask]
        features_used = [feature_names[i] for i in feature_mask]
    else:
        X_used = X.copy()
        features_used = feature_names.copy()

    unique_batches = np.unique(batches)
    predictions = []
    actuals = []
    pc_explained_var = None

    # Leave-One-Batch-Out Cross-Validation
    for test_batch in unique_batches:
        test_mask = batches == test_batch
        train_mask = ~test_mask

        X_train, X_test = X_used[train_mask], X_used[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        batch_train = batches[train_mask]

        if len(X_train) < 5 or len(X_test) == 0:
            continue

        # Standardize (fit on train only)
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        # PCA (fit on train only)
        pca = PCA()
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)

        # Store explained variance (first fold only)
        if pc_explained_var is None:
            pc_explained_var = pca.explained_variance_ratio_

        # Keep only PC1 and PC2
        X_train_pca_reduced = X_train_pca.copy()
        X_test_pca_reduced = X_test_pca.copy()
        X_train_pca_reduced[:, 2:] = 0
        X_test_pca_reduced[:, 2:] = 0

        # Inverse transform back to original space
        X_train_reconstructed = pca.inverse_transform(X_train_pca_reduced)
        X_test_reconstructed = pca.inverse_transform(X_test_pca_reduced)

        # Inverse standardization
        X_train_inv = scaler.inverse_transform(X_train_reconstructed)
        X_test_inv = scaler.inverse_transform(X_test_reconstructed)

        # === Common Slope計算（論文の手法）===
        # 訓練セット内の各バッチから傾きを計算
        train_unique_batches = np.unique(batch_train)
        batch_slopes = []

        for batch_id in train_unique_batches:
            batch_mask = batch_train == batch_id
            if np.sum(batch_mask) > 1:
                X_batch = X_train_inv[batch_mask]
                y_batch = y[train_mask][batch_mask]  # 元のyを使用

                # logSigma (features_usedの中での位置を確認)
                if 'logSigma' in features_used:
                    logSigma_idx = features_used.index('logSigma')
                    # logSigmaとlogNの回帰
                    model_batch = LinearRegression()
                    model_batch.fit(X_batch[:, logSigma_idx].reshape(-1, 1), y_batch)
                    batch_slopes.append(model_batch.coef_[0])

        # 平均傾きを計算
        if len(batch_slopes) > 0:
            alpha_common = np.mean(batch_slopes)
        else:
            alpha_common = -3.0  # デフォルト値（S-N曲線の典型的な傾き）

        # === テストセットの予測 ===
        # point_i: テストセットの最後の点（論文に従う）
        if 'logSigma' in features_used:
            logSigma_idx = features_used.index('logSigma')
            
            # テストセットの1点を使って切片を調整
            point_i_logSigma = X_test_inv[-1, logSigma_idx]
            point_i_logN = y_test[-1]
            
            # 切片の計算: beta = logN_i - alpha * logSigma_i
            beta_adj = point_i_logN - alpha_common * point_i_logSigma
            
            # Common Slopeを使った予測
            y_pred = alpha_common * X_test_inv[:, logSigma_idx] + beta_adj
        else:
            # logSigmaが特徴に含まれていない場合は通常の回帰
            if model_type == 'linear':
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_inv, y_train)
            y_pred = model.predict(X_test_inv)

        predictions.extend(y_pred)
        actuals.extend(y_test)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate metrics
    rse, r2, mae = calculate_metrics(actuals, predictions)

    # Feature ranking (train on full data)
    scaler_full = StandardScaler()
    X_full_std = scaler_full.fit_transform(X_used)
    pca_full = PCA()
    X_full_pca = pca_full.fit_transform(X_full_std)
    X_full_pca_reduced = X_full_pca.copy()
    X_full_pca_reduced[:, 2:] = 0
    X_full_reconstructed = pca_full.inverse_transform(X_full_pca_reduced)
    X_full_inv = scaler_full.inverse_transform(X_full_reconstructed)

    if model_type == 'linear':
        model_full = LinearRegression()
    else:
        model_full = RandomForestRegressor(n_estimators=100, random_state=42)

    model_full.fit(X_full_inv, y)
    ranking = get_feature_importance(model_full, features_used, model_type)

    return rse, r2, mae, ranking, pc_explained_var

# ===================================================================
# PCA_PAPER METHOD (from main_16.py)
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

# def pca_common_loocv_with_bootstrap(X, y, batches, feature_names, n_iterations=100):
    """
    Stage 2: 100回繰り返しサンプリングを追加
    """
    all_rse = []
    all_r2 = []
    all_mae = []
    
    np.random.seed(42)  # 再現性
    
    for iter in range(n_iterations):
        # 各バッチから5サンプルをランダムサンプリング（復元抽出）
        subset_indices = []
        for batch_id in np.unique(batches):
            batch_mask = (batches == batch_id)
            batch_indices = np.where(batch_mask)[0]
            
            # 復元抽出で5個サンプリング
            sampled = np.random.choice(batch_indices, size=5, replace=True)
            subset_indices.extend(sampled)
        
        # サブセット作成
        subset_indices = np.array(subset_indices)
        X_subset = X[subset_indices]
        y_subset = y[subset_indices]
        batches_subset = batches[subset_indices]
        
        # 修正版LOOCVを実行
        rse, r2, mae, ranking, pc_var = pca_common_loocv(
            X_subset, y_subset, batches_subset, feature_names, model_type='linear'
        )
        
        all_rse.append(rse)
        all_r2.append(r2)
        all_mae.append(mae)
        
        if (iter + 1) % 10 == 0:
            print(f"Iteration {iter+1}/{n_iterations}: RSE={rse:.4f}, R²={r2:.4f}")
    
    # 統計量を計算
    mean_rse = np.mean(all_rse)
    std_rse = np.std(all_rse)
    mean_r2 = np.mean(all_r2)
    std_r2 = np.std(all_r2)
    
    print(f"\n最終結果（100回平均）:")
    print(f"  RSE: {mean_rse:.4f} ± {std_rse:.4f}")
    print(f"  R²:  {mean_r2:.4f} ± {std_r2:.4f}")
    
    return mean_rse, mean_r2, mean_mae, all_rse, all_r2, all_mae



# def feature_selection_loocv(X, y, batches, feature_names, method='spearman', model_type='linear', exclude_feature=None):
    """Feature selection methods (Spearman/HGVS) with LOOCV"""
    unique_batches = np.unique(batches)
    predictions = []
    actuals = []

    # Leave-One-Batch-Out Cross-Validation
    for test_batch in unique_batches:
        test_mask = batches == test_batch
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(X_train) < 10 or len(X_test) == 0:
            continue

        # Shuffle training data
        # shuffle_idx = np.random.permutation(len(X_train))
        # X_train = X_train[shuffle_idx]
        # y_train = y_train[shuffle_idx]

        # Select top 4 features
        if method == 'spearman':
            top_indices, top_features = spearman_feature_selection(X_train, y_train, feature_names, top_k=4)
        else:  # HGVS
            top_indices, top_features = hgvs_feature_selection(X_train, y_train, feature_names, top_k=4)

        # Exclude feature if specified
        if exclude_feature is not None and exclude_feature in top_features:
            top_features_used = [f for f in top_features if f != exclude_feature]
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

    # Feature ranking (on full data)
    if method == 'spearman':
        top_indices_full, top_features_full = spearman_feature_selection(X, y, feature_names, top_k=4)
    else:
        top_indices_full, top_features_full = hgvs_feature_selection(X, y, feature_names, top_k=4)

    if exclude_feature is not None and exclude_feature in top_features_full:
        top_features_full = [f for f in top_features_full if f != exclude_feature]
        top_indices_full = [feature_names.index(f) for f in top_features_full]

    X_selected_full = X[:, top_indices_full]

    if model_type == 'linear':
        model_full = LinearRegression()
    else:
        model_full = RandomForestRegressor(n_estimators=100, random_state=42)

    model_full.fit(X_selected_full, y)
    ranking = get_feature_importance(model_full, top_features_full, model_type)

    return rse, r2, mae, ranking

# ===================================================================
# NORMAL TRAIN-TEST SPLIT METHODS
# ===================================================================

# def pca_common_normal(X, y, batches, feature_names, model_type='linear', exclude_feature=None, test_size=0.2):
    """PCA Common with normal train-test split"""
    if exclude_feature is not None:
        exclude_idx = feature_names.index(exclude_feature)
        feature_mask = [i for i in range(len(feature_names)) if i != exclude_idx]
        X_used = X[:, feature_mask]
        features_used = [feature_names[i] for i in feature_mask]
    else:
        X_used = X.copy()
        features_used = feature_names.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_used, y, test_size=test_size, random_state=42, shuffle=True
    )

    # Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # PCA
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    pc_explained_var = pca.explained_variance_ratio_

    # Keep only PC1 and PC2
    X_train_pca_reduced = X_train_pca.copy()
    X_test_pca_reduced = X_test_pca.copy()
    X_train_pca_reduced[:, 2:] = 0
    X_test_pca_reduced[:, 2:] = 0

    # Inverse transform
    X_train_reconstructed = pca.inverse_transform(X_train_pca_reduced)
    X_test_reconstructed = pca.inverse_transform(X_test_pca_reduced)
    X_train_inv = scaler.inverse_transform(X_train_reconstructed)
    X_test_inv = scaler.inverse_transform(X_test_reconstructed)

    # Train model
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train_inv, y_train)
    y_pred = model.predict(X_test_inv)

    # Calculate metrics
    rse, r2, mae = calculate_metrics(y_test, y_pred)
    ranking = get_feature_importance(model, features_used, model_type)

    return rse, r2, mae, ranking
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


def pca_common_normal(X, y, batches, feature_names, model_type='linear', exclude_feature=None, test_size=0.2):
    """PCA Common with normal train-test split (Common Slope適用)"""
    if exclude_feature is not None:
        exclude_idx = feature_names.index(exclude_feature)
        feature_mask = [i for i in range(len(feature_names)) if i != exclude_idx]
        X_used = X[:, feature_mask]
        features_used = [feature_names[i] for i in feature_mask]
    else:
        X_used = X.copy()
        features_used = feature_names.copy()

    X_train, X_test, y_train, y_test, batch_train, batch_test = train_test_split(
        X_used, y, batches, test_size=test_size, random_state=42, shuffle=True
    )

    # Standardize
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # PCA
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)

    # Keep only PC1 and PC2
    X_train_pca_reduced = X_train_pca.copy()
    X_test_pca_reduced = X_test_pca.copy()
    X_train_pca_reduced[:, 2:] = 0
    X_test_pca_reduced[:, 2:] = 0

    # Inverse transform
    X_train_reconstructed = pca.inverse_transform(X_train_pca_reduced)
    X_test_reconstructed = pca.inverse_transform(X_test_pca_reduced)
    X_train_inv = scaler.inverse_transform(X_train_reconstructed)
    X_test_inv = scaler.inverse_transform(X_test_reconstructed)

    # Common Slope calculation
    if 'logSigma' in features_used:
        logSigma_idx = features_used.index('logSigma')
        
        train_unique_batches = np.unique(batch_train)
        batch_slopes = []
        
        for batch_id in train_unique_batches:
            batch_mask = batch_train == batch_id
            if np.sum(batch_mask) > 1:
                X_batch = X_train_inv[batch_mask]
                y_batch = y_train[batch_mask]
                
                model_batch = LinearRegression()
                model_batch.fit(X_batch[:, logSigma_idx].reshape(-1, 1), y_batch)
                batch_slopes.append(model_batch.coef_[0])
        
        if len(batch_slopes) > 0:
            alpha_common = np.mean(batch_slopes)
        else:
            alpha_common = -3.0
        
        # Use last training point for intercept
        point_i_logSigma = X_train_inv[-1, logSigma_idx]
        point_i_logN = y_train[-1]
        beta_adj = point_i_logN - alpha_common * point_i_logSigma
        
        # Predict
        y_pred = alpha_common * X_test_inv[:, logSigma_idx] + beta_adj
    else:
        # Fallback to standard regression
        if model_type == 'linear':
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_inv, y_train)
        y_pred = model.predict(X_test_inv)

    # Calculate metrics
    rse, r2, mae = calculate_metrics(y_test, y_pred)
    
    # Feature ranking
    if model_type == 'linear':
        model_full = LinearRegression()
    else:
        model_full = RandomForestRegressor(n_estimators=100, random_state=42)
    model_full.fit(X_train_inv, y_train)
    ranking = get_feature_importance(model_full, features_used, model_type)

    return rse, r2, mae, ranking


def pca_paper_normal(X, y, batches, feature_names, model_type='linear', exclude_feature=None, test_size=0.2):
    """PCA Paper with normal train-test split"""
    if exclude_feature is not None:
        exclude_idx = feature_names.index(exclude_feature)
        feature_mask = [i for i in range(len(feature_names)) if i != exclude_idx]
        X_used = X[:, feature_mask]
        features_used = [feature_names[i] for i in feature_mask]
    else:
        X_used = X.copy()
        features_used = feature_names.copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_used, y, test_size=test_size, random_state=42, shuffle=True
    )

    # PCA on all data including target
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    X_with_target = np.column_stack([X_all, y_all])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_target)

    pca = PCA(n_components=min(6, X_with_target.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Keep only PC1 and PC2
    X_pca_reduced = X_pca.copy()
    X_pca_reduced[:, 2:] = 0
    X_reconstructed = pca.inverse_transform(X_pca_reduced)
    X_inv = scaler.inverse_transform(X_reconstructed)

    # Split back
    X_train_inv = X_inv[:len(X_train), :-1]
    X_test_inv = X_inv[len(X_train):, :-1]

    # Train model
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train_inv, y_train)
    y_pred = model.predict(X_test_inv)

    # Calculate metrics
    rse, r2, mae = calculate_metrics(y_test, y_pred)
    ranking = get_feature_importance(model, features_used, model_type)

    return rse, r2, mae, ranking

def feature_selection_normal(X, y, batches, feature_names, method='spearman', model_type='linear', exclude_feature=None, test_size=0.2):
    """Feature selection with normal train-test split"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

    # Shuffle training data
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    # Select top 4 features
    if method == 'spearman':
        top_indices, top_features = spearman_feature_selection(X_train, y_train, feature_names, top_k=4)
    else:
        top_indices, top_features = hgvs_feature_selection(X_train, y_train, feature_names, top_k=4)

    if exclude_feature is not None and exclude_feature in top_features:
        top_features = [f for f in top_features if f != exclude_feature]
        top_indices = [feature_names.index(f) for f in top_features]

    X_train_selected = X_train[:, top_indices]
    X_test_selected = X_test[:, top_indices]

    # Train model
    if model_type == 'linear':
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    # Calculate metrics
    rse, r2, mae = calculate_metrics(y_test, y_pred)
    ranking = get_feature_importance(model, top_features, model_type)

    return rse, r2, mae, ranking

# ===================================================================
# MAIN EXECUTION
# ===================================================================

# Create output directory
now = datetime.now()
dir_name = f"result_{now.strftime('%m%d_%H%M')}"
os.makedirs(dir_name, exist_ok=True)
print(f"\nOutput directory created: {dir_name}")

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

# PCA
print("\nProcessing PCA Common (Linear LOOCV)...")
full_rse, full_r2, full_mae, full_ranking, pc_var = pca_common_loocv(X, y, batches, feature_names, model_type='linear')
print(f"  PC1 variance: {pc_var[0]:.4f}, PC2 variance: {pc_var[1]:.4f}")
drop1_rse, drop1_r2, drop1_mae, drop1_ranking, _ = pca_common_loocv(X, y, batches, feature_names, model_type='linear', exclude_feature=full_ranking[0])
results_loocv_linear.append({
    'Method': 'PCA',
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
df_loocv_linear.to_csv(f'{dir_name}/LOOCV_Linear.csv', index=False)
print(f"\nSaved: {dir_name}/LOOCV_Linear.csv")

# ==================== LOOCV WITH RANDOM FOREST ====================
print("\n" + "="*70)
print("RUNNING LOOCV WITH RANDOM FOREST")
print("="*70)

results_loocv_random = []

# PCA_paper
print("\nProcessing PCA_paper (Random Forest LOOCV)...")
full_rse, full_r2, full_mae, full_ranking, _ = pca_paper_loocv(X, y, batches, feature_names, model_type='rf')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking, _ = pca_paper_loocv(X, y, batches, feature_names, model_type='rf', exclude_feature=full_ranking[0])
results_loocv_random.append({
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

# PCA
print("\nProcessing PCA Common (Random Forest LOOCV)...")
full_rse, full_r2, full_mae, full_ranking, _ = pca_common_loocv(X, y, batches, feature_names, model_type='rf')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking, _ = pca_common_loocv(X, y, batches, feature_names, model_type='rf', exclude_feature=full_ranking[0])
results_loocv_random.append({
    'Method': 'PCA',
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
print("\nProcessing Spearman (Random Forest LOOCV)...")
full_rse, full_r2, full_mae, full_ranking = feature_selection_loocv(X, y, batches, feature_names, method='spearman', model_type='rf')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = feature_selection_loocv(X, y, batches, feature_names, method='spearman', model_type='rf', exclude_feature=full_ranking[0])
results_loocv_random.append({
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
print("\nProcessing HGVS (Random Forest LOOCV)...")
full_rse, full_r2, full_mae, full_ranking = feature_selection_loocv(X, y, batches, feature_names, method='hgvs', model_type='rf')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = feature_selection_loocv(X, y, batches, feature_names, method='hgvs', model_type='rf', exclude_feature=full_ranking[0])
results_loocv_random.append({
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

df_loocv_random = pd.DataFrame(results_loocv_random)
df_loocv_random.to_csv(f'{dir_name}/LOOCV_Random.csv', index=False)
print(f"\nSaved: {dir_name}/LOOCV_Random.csv")

# ==================== NORMAL SPLIT WITH LINEAR REGRESSION ====================
print("\n" + "="*70)
print("RUNNING NORMAL TRAIN-TEST SPLIT WITH LINEAR REGRESSION")
print("="*70)

results_normal_linear = []

# PCA_paper
print("\nProcessing PCA_paper (Linear Normal)...")
full_rse, full_r2, full_mae, full_ranking = pca_paper_normal(X, y, batches, feature_names, model_type='linear')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = pca_paper_normal(X, y, batches, feature_names, model_type='linear', exclude_feature=full_ranking[0])
results_normal_linear.append({
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

# PCA
print("\nProcessing PCA Common (Linear Normal)...")
full_rse, full_r2, full_mae, full_ranking = pca_common_normal(X, y, batches, feature_names, model_type='linear')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = pca_common_normal(X, y, batches, feature_names, model_type='linear', exclude_feature=full_ranking[0])
results_normal_linear.append({
    'Method': 'PCA',
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
print("\nProcessing Spearman (Linear Normal)...")
full_rse, full_r2, full_mae, full_ranking = feature_selection_normal(X, y, batches, feature_names, method='spearman', model_type='linear')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = feature_selection_normal(X, y, batches, feature_names, method='spearman', model_type='linear', exclude_feature=full_ranking[0])
results_normal_linear.append({
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
print("\nProcessing HGVS (Linear Normal)...")
full_rse, full_r2, full_mae, full_ranking = feature_selection_normal(X, y, batches, feature_names, method='hgvs', model_type='linear')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = feature_selection_normal(X, y, batches, feature_names, method='hgvs', model_type='linear', exclude_feature=full_ranking[0])
results_normal_linear.append({
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

df_normal_linear = pd.DataFrame(results_normal_linear)
df_normal_linear.to_csv(f'{dir_name}/Normal_Linear.csv', index=False)
print(f"\nSaved: {dir_name}/Normal_Linear.csv")

# ==================== NORMAL SPLIT WITH RANDOM FOREST ====================
print("\n" + "="*70)
print("RUNNING NORMAL TRAIN-TEST SPLIT WITH RANDOM FOREST")
print("="*70)

results_normal_random = []

# PCA_paper
print("\nProcessing PCA_paper (Random Forest Normal)...")
full_rse, full_r2, full_mae, full_ranking = pca_paper_normal(X, y, batches, feature_names, model_type='rf')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = pca_paper_normal(X, y, batches, feature_names, model_type='rf', exclude_feature=full_ranking[0])
results_normal_random.append({
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

# PCA
print("\nProcessing PCA Common (Random Forest Normal)...")
full_rse, full_r2, full_mae, full_ranking = pca_common_normal(X, y, batches, feature_names, model_type='rf')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = pca_common_normal(X, y, batches, feature_names, model_type='rf', exclude_feature=full_ranking[0])
results_normal_random.append({
    'Method': 'PCA',
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
print("\nProcessing Spearman (Random Forest Normal)...")
full_rse, full_r2, full_mae, full_ranking = feature_selection_normal(X, y, batches, feature_names, method='spearman', model_type='rf')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = feature_selection_normal(X, y, batches, feature_names, method='spearman', model_type='rf', exclude_feature=full_ranking[0])
results_normal_random.append({
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
print("\nProcessing HGVS (Random Forest Normal)...")
full_rse, full_r2, full_mae, full_ranking = feature_selection_normal(X, y, batches, feature_names, method='hgvs', model_type='rf')
drop1_rse, drop1_r2, drop1_mae, drop1_ranking = feature_selection_normal(X, y, batches, feature_names, method='hgvs', model_type='rf', exclude_feature=full_ranking[0])
results_normal_random.append({
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

df_normal_random = pd.DataFrame(results_normal_random)
df_normal_random.to_csv(f'{dir_name}/Normal_Random.csv', index=False)
print(f"\nSaved: {dir_name}/Normal_Random.csv")

print("\n" + "="*70)
print("ALL PROCESSING COMPLETED")
print(f"Results saved in directory: {dir_name}")
print("="*70)
