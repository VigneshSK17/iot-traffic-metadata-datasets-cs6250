import os
import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def load_processed_data(filepath: str) -> pl.DataFrame:
    """Loads processed parquet dataset using Polars for optimized performance."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Ensure your preprocessing script has run."
        )
    print(f"Loading dataset from {filepath}...")
    return pl.read_parquet(filepath)

def prepare_features(df: pl.DataFrame, target_col: str):
    """
    Separates the Polars DataFrame into feature matrix (X), target array (y), 
    and optional metadata arrays.
    """
    drop_cols = [target_col, "device_id", "device_name", "flow_id", "timestamp", "ip_src", "ip_dst"]
    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in actual_drop_cols]
    
    X_raw = df.select(feature_cols).to_pandas()
    X = X_raw.select_dtypes(include=[np.number])
    y = df.select(target_col).to_pandas().values.ravel()
    
    return X, y

def print_top_features(model, feature_names, top_n=5):
    """Helper function to cleanly display top N feature importances."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n--- Top {top_n} Most Informative Features ---")
    for i in sorted_idx[:top_n]:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")

def align_features(X_train, X_test):
    """
    Restricts both train and test sets to the exact intersection of feature columns 
    present in both datasets to ensure identical dimensions.
    """
    common_cols = [col for col in X_train.columns if col in X_test.columns]
    print(f"Feature Alignment: Found {len(common_cols)} overlapping features between train and test datasets.")
    
    X_train_aligned = X_train[common_cols].copy()
    X_test_aligned = X_test[common_cols].copy()
    
    return X_train_aligned, X_test_aligned

def main():
    # --- Configuration ---
    target_column = "device_type"
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # PHASE 1: INTRA-DATASET EVALUATION (Dataset A: CIC IoT 2022)
    # -------------------------------------------------------------------------
    cic_path = "../processed/cic_iot_dataset2022.parquet"
    df_cic = load_processed_data(cic_path)
    
    if target_column not in df_cic.columns:
        raise ValueError(f"Target column '{target_column}' not found in CIC dataset.")
        
    X_cic, y_cic = prepare_features(df_cic, target_column)
    print(f"CIC Dataset loaded: {X_cic.shape[0]} samples, {X_cic.shape[1]} numeric features.")

    # =========================================================================
    # RUN 1: CIC RANDOM BASELINE
    # =========================================================================
    print("\n" + "="*70)
    print("  RUN 1: CIC RANDOM SPLIT (Baseline)")
    print("  [Context: Random 80/20 split on CIC. Tests instance memorization.]")
    print("="*70)
    
    X_tr_r1, X_te_r1, y_tr_r1, y_te_r1 = train_test_split(X_cic, y_cic, test_size=0.2, random_state=42, stratify=y_cic)
    rf_r1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_r1.fit(X_tr_r1, y_tr_r1)
    acc_r1 = accuracy_score(y_te_r1, rf_r1.predict(X_te_r1))
    print(f"Overall Accuracy: {acc_r1 * 100:.2f}%")
    print_top_features(rf_r1, X_cic.columns)
    joblib.dump(rf_r1, os.path.join(output_dir, "rf_run1_cic_random.joblib"))

    # =========================================================================
    # RUN 2: CIC UNSEEN HARDWARE SPLIT (Grouped by device_id)
    # =========================================================================
    print("\n" + "="*70)
    print("  RUN 2: CIC UNSEEN HARDWARE SPLIT (Grouped by device_id)")
    print("  [Context: Leave-One-Device-Out split on CIC. Tests unseen hardware.]")
    print("="*70)
    
    if "device_id" in df_cic.columns:
        groups_cic = df_cic.select("device_id").to_pandas().values.ravel()
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(gss.split(X_cic, y_cic, groups=groups_cic))
        
        X_tr_r2, X_te_r2 = X_cic.iloc[tr_idx], X_cic.iloc[te_idx]
        y_tr_r2, y_te_r2 = y_cic[tr_idx], y_cic[te_idx]
        
        rf_r2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_r2.fit(X_tr_r2, y_tr_r2)
        acc_r2 = accuracy_score(y_te_r2, rf_r2.predict(X_te_r2))
        print(f"Overall Accuracy: {acc_r2 * 100:.2f}%")
        print_top_features(rf_r2, X_cic.columns)
        joblib.dump(rf_r2, os.path.join(output_dir, "rf_run2_cic_unseen_device.joblib"))

    # =========================================================================
    # RUN 3: CIC CHRONOLOGICAL TIME-WINDOW SPLIT
    # =========================================================================
    print("\n" + "="*70)
    print("  RUN 3: CIC CHRONOLOGICAL SPLIT (Past to Future)")
    print("  [Context: Past 80% train, future 20% test on CIC. Tests temporal drift.]")
    print("="*70)
    
    if "epoch_timestamp" in df_cic.columns:
        timestamps = df_cic.select("epoch_timestamp").to_pandas().values.ravel()
        sorted_idx = np.argsort(timestamps)
        split_idx = int(len(sorted_idx) * 0.8)
        
        tr_time_idx, te_time_idx = sorted_idx[:split_idx], sorted_idx[split_idx:]
        X_tr_r3, X_te_r3 = X_cic.iloc[tr_time_idx], X_cic.iloc[te_time_idx]
        y_tr_r3, y_te_r3 = y_cic[tr_time_idx], y_cic[te_time_idx]
        
        rf_r3 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_r3.fit(X_tr_r3, y_tr_r3)
        acc_r3 = accuracy_score(y_te_r3, rf_r3.predict(X_te_r3))
        print(f"Overall Accuracy: {acc_r3 * 100:.2f}%")
        print_top_features(rf_r3, X_cic.columns)
        joblib.dump(rf_r3, os.path.join(output_dir, "rf_run3_cic_chronological.joblib"))

    # -------------------------------------------------------------------------
    # PHASE 2: CROSS-DATASET TRANSFER EVALUATION (Train CIC -> Test UNSW)
    # -------------------------------------------------------------------------
    unsw_path = "../processed/unsw_iot_dataset.parquet"
    if os.path.exists(unsw_path):
        df_unsw = load_processed_data(unsw_path)
        target_column_unsw = target_column if target_column in df_unsw.columns else df_unsw.columns[-1]
            
        X_unsw_raw, y_unsw = prepare_features(df_unsw, target_column_unsw)
        
        common_classes = set(np.unique(y_cic))
        mask = np.isin(y_unsw, list(common_classes))
        X_unsw_filtered = X_unsw_raw.iloc[mask]
        y_unsw_filtered = y_unsw[mask]
        
        print(f"\nUNSW Dataset loaded: {X_unsw_raw.shape[0]} samples total ({X_unsw_filtered.shape[0]} samples retained in common classes).")

        # =====================================================================
        # RUN 4: FULL CROSS-DATASET TRANSFER
        # =====================================================================
        print("\n" + "="*70)
        print("  RUN 4: CROSS-DATASET TRANSFER (Full Train CIC -> Test UNSW)")
        print("  [Context: Trained on 100% of CIC, tested on UNSW. Domain transfer baseline.]")
        print("="*70)
        
        X_tr_r4, X_te_r4 = align_features(X_cic, X_unsw_filtered)
        rf_r4 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_r4.fit(X_tr_r4.values, y_cic)
        
        acc_r4 = accuracy_score(y_unsw_filtered, rf_r4.predict(X_te_r4.values))
        print(f"Overall Cross-Dataset Accuracy: {acc_r4 * 100:.2f}%")
        print_top_features(rf_r4, X_tr_r4.columns)
        joblib.dump(rf_r4, os.path.join(output_dir, "rf_run4_cross_full.joblib"))

        # =====================================================================
        # RUN 5: GROUP-FILTERED CROSS-DATASET TRANSFER
        # =====================================================================
        print("\n" + "="*70)
        print("  RUN 5: CROSS-DATASET TRANSFER (Train on Device Subset -> Test UNSW)")
        print("  [Context: Trains on a grouped subset of CIC devices, tests on UNSW.]")
        print("="*70)
        
        if "device_id" in df_cic.columns:
            unique_devices = np.unique(groups_cic)
            half_devices = unique_devices[:len(unique_devices)//2]
            subset_mask = np.isin(groups_cic, half_devices)
            
            X_tr_r5_raw = X_cic.iloc[subset_mask]
            y_tr_r5 = y_cic[subset_mask]
            
            X_tr_r5, X_te_r5 = align_features(X_tr_r5_raw, X_unsw_filtered)
            rf_r5 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_r5.fit(X_tr_r5.values, y_tr_r5)
            
            acc_r5 = accuracy_score(y_unsw_filtered, rf_r5.predict(X_te_r5.values))
            print(f"Overall Cross-Dataset Accuracy (Device Subset): {acc_r5 * 100:.2f}%")
            print_top_features(rf_r5, X_tr_r5.columns)
            joblib.dump(rf_r5, os.path.join(output_dir, "rf_run5_cross_device_subset.joblib"))

        # =====================================================================
        # RUN 6: CHRONOLOGICALLY FILTERED CROSS-DATASET TRANSFER
        # =====================================================================
        print("\n" + "="*70)
        print("  RUN 6: CROSS-DATASET TRANSFER (Train on Early CIC -> Test UNSW)")
        print("  [Context: Trains only on early chronological CIC traffic, tests on UNSW.]")
        print("="*70)
        
        if "epoch_timestamp" in df_cic.columns:
            early_train_idx = sorted_idx[:split_idx]
            X_tr_r6_raw = X_cic.iloc[early_train_idx]
            y_tr_r6 = y_cic[early_train_idx]
            
            X_tr_r6, X_te_r6 = align_features(X_tr_r6_raw, X_unsw_filtered)
            rf_r6 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf_r6.fit(X_tr_r6.values, y_tr_r6)
            
            acc_r6 = accuracy_score(y_unsw_filtered, rf_r6.predict(X_te_r6.values))
            print(f"Overall Cross-Dataset Accuracy (Early Training Window): {acc_r6 * 100:.2f}%")
            print_top_features(rf_r6, X_tr_r6.columns)
            joblib.dump(rf_r6, os.path.join(output_dir, "rf_run6_cross_chronological.joblib"))
            print("="*70)
    else:
        print("Skipping Runs 4, 5, & 6: UNSW dataset not found in ../processed/.")

if __name__ == "__main__":
    main()
