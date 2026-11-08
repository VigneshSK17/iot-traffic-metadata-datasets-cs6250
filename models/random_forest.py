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

def prepare_data(df: pl.DataFrame, target_col: str, group_col: str):
    """
    Separates the Polars DataFrame into feature matrix (X), target array (y), 
    and group array (groups) for group-based splitting.
    """
    drop_cols = [target_col, group_col, "flow_id", "timestamp", "ip_src", "ip_dst", "device_name"]
    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in actual_drop_cols]
    
    # Convert Polars selections to Pandas
    X_raw = df.select(feature_cols).to_pandas()
    
    # Keep only numeric features
    X = X_raw.select_dtypes(include=[np.number])
    y = df.select(target_col).to_pandas().values.ravel()
    
    groups = None
    if group_col in df.columns:
        groups = df.select(group_col).to_pandas().values.ravel()
    else:
        print(f"Warning: Group column '{group_col}' not found in dataset.")
        
    print(f"Selected {X.shape[1]} numeric feature columns for training.")
    return X, y, groups

def print_top_features(model, feature_names, top_n=5):
    """Helper function to cleanly display top N feature importances for any model."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n--- Top {top_n} Most Informative Features ---")
    for i in sorted_idx[:top_n]:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")

def main():
    # --- Configuration ---
    dataset_filename = "cic_iot_dataset2022.parquet"
    data_path = f"../processed/{dataset_filename}"
    target_column = "device_type" 
    group_column = "device_id"
    
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    df = load_processed_data(data_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {dataset_filename}.")
    
    # 2. Preprocess and Extract Groups
    X, y, groups = prepare_data(df, target_col=target_column, group_col=group_column)
    
    # Identify temporal/timestamp columns to easily strip when needed
    time_cols = [col for col in X.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
    print(f"Identified temporal features: {time_cols}")

    # =========================================================================
    # RUN 1: BASELINE (Standard Random Split - With Timestamps)
    # =========================================================================
    print("\n" + "="*70)
    print("  RUN 1: BASELINE (Standard Random Split - With Timestamps)")
    print("  [Context: Randomly shuffles flows. Tests general classification power")
    print("   when instances of the exact same devices appear in train and test sets.]")
    print("="*70)
    
    X_train_r1, X_test_r1, y_train_r1, y_test_r1 = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf_r1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_r1.fit(X_train_r1, y_train_r1)
    
    acc_r1 = accuracy_score(y_test_r1, rf_r1.predict(X_test_r1))
    print(f"Overall Accuracy: {acc_r1 * 100:.2f}%")
    print_top_features(rf_r1, X.columns)
    joblib.dump(rf_r1, os.path.join(output_dir, "rf_run1_baseline.joblib"))

    # =========================================================================
    # RUN 2: BASELINE W/O TIMESTAMPS (Standard Random Split - No Timestamps)
    # =========================================================================
    print("\n" + "="*70)
    print("  RUN 2: BASELINE W/O TIMESTAMPS (Standard Random Split - No Timestamps)")
    print("  [Context: Same random split as Run 1, but strips timestamps to check")
    print("   if temporal capture leakage was driving the high score.]")
    print("="*70)
    
    X_r2 = X.drop(columns=time_cols, errors='ignore')
    X_train_r2, X_test_r2, y_train_r2, y_test_r2 = train_test_split(
        X_r2, y, test_size=0.2, random_state=42, stratify=y
    )
    rf_r2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_r2.fit(X_train_r2, y_train_r2)
    
    acc_r2 = accuracy_score(y_test_r2, rf_r2.predict(X_test_r2))
    print(f"Overall Accuracy: {acc_r2 * 100:.2f}%")
    print_top_features(rf_r2, X_r2.columns)
    joblib.dump(rf_r2, os.path.join(output_dir, "rf_run2_baseline_notime.joblib"))

    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        # =====================================================================
        # RUN 3: UNSEEN DEVICES (Group-Based Split - With Timestamps)
        # =====================================================================
        print("\n" + "="*70)
        print("  RUN 3: UNSEEN DEVICES (Group-Based Split - With Timestamps)")
        print("  [Context: Splits by device_id. Tests whether the model can recognize")
        print("   device types on brand-new physical hardware it has never seen before.]")
        print("="*70)
        
        X_train_r3, X_test_r3 = X.iloc[train_idx], X.iloc[test_idx]
        y_train_r3, y_test_r3 = y[train_idx], y[test_idx]
        rf_r3 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_r3.fit(X_train_r3, y_train_r3)
        
        acc_r3 = accuracy_score(y_test_r3, rf_r3.predict(X_test_r3))
        print(f"Overall Accuracy: {acc_r3 * 100:.2f}%")
        print_top_features(rf_r3, X.columns)
        joblib.dump(rf_r3, os.path.join(output_dir, "rf_run3_unseen.joblib"))

        # =====================================================================
        # RUN 4: UNSEEN DEVICES W/O TIMESTAMPS (Group-Based Split - No Timestamps)
        # =====================================================================
        print("\n" + "="*70)
        print("  RUN 4: UNSEEN DEVICES W/O TIMESTAMPS (Group-Based Split - No Timestamps)")
        print("  [Context: Device-grouped split with timestamps removed to isolate")
        print("   if hardware generalization fails due to clock timing or network behavior.]")
        print("="*70)
        
        X_r4 = X.drop(columns=time_cols, errors='ignore')
        X_train_r4, X_test_r4 = X_r4.iloc[train_idx], X_r4.iloc[test_idx]
        y_train_r4, y_test_r4 = y[train_idx], y[test_idx]
        rf_r4 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_r4.fit(X_train_r4, y_train_r4)
        
        acc_r4 = accuracy_score(y_test_r4, rf_r4.predict(X_test_r4))
        print(f"Overall Accuracy: {acc_r4 * 100:.2f}%")
        print_top_features(rf_r4, X_r4.columns)
        joblib.dump(rf_r4, os.path.join(output_dir, "rf_run4_unseen_notime.joblib"))

    # =========================================================================
    # RUNS 5 & 6: CHRONOLOGICAL (TIME-WINDOW) SPLITS
    # =========================================================================
    primary_time_col = next((col for col in time_cols if 'epoch' in col.lower() or 'timestamp' in col.lower()), None)
    
    if primary_time_col and primary_time_col in X.columns:
        print(f"\nSorting data chronologically using '{primary_time_col}' for time-window splits...")
        sorted_indices = np.argsort(X[primary_time_col].values)
        split_idx = int(len(sorted_indices) * 0.8)
        train_time_idx = sorted_indices[:split_idx]
        test_time_idx = sorted_indices[split_idx:]
        
        # RUN 5: Chronological - With Timestamps
        print("\n" + "="*70)
        print("  RUN 5: CHRONOLOGICAL SPLIT (Time-Window - With Timestamps)")
        print("  [Context: Trains on past traffic (first 80%) and tests on future traffic")
        print("   (final 20%) to simulate real-world deployment over time.]")
        print("="*70)
        
        X_train_r5, X_test_r5 = X.iloc[train_time_idx], X.iloc[test_time_idx]
        y_train_r5, y_test_r5 = y[train_time_idx], y[test_time_idx]
        rf_r5 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_r5.fit(X_train_r5, y_train_r5)
        
        acc_r5 = accuracy_score(y_test_r5, rf_r5.predict(X_test_r5))
        print(f"Overall Accuracy: {acc_r5 * 100:.2f}%")
        print_top_features(rf_r5, X.columns)
        joblib.dump(rf_r5, os.path.join(output_dir, "rf_run5_chronological.joblib"))

        # RUN 6: Chronological - Without Timestamps
        print("\n" + "="*70)
        print("  RUN 6: CHRONOLOGICAL SPLIT (Time-Window - No Timestamps)")
        print("  [Context: Chronological past-to-future split with timestamps removed")
        print("   to test if traffic behavior drifts or holds stable over time.]")
        print("="*70)
        
        X_r6 = X.drop(columns=time_cols, errors='ignore')
        X_train_r6, X_test_r6 = X_r6.iloc[train_time_idx], X_r6.iloc[test_time_idx]
        y_train_r6, y_test_r6 = y[train_time_idx], y[test_time_idx]
        rf_r6 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_r6.fit(X_train_r6, y_train_r6)
        
        acc_r6 = accuracy_score(y_test_r6, rf_r6.predict(X_test_r6))
        print(f"Overall Accuracy: {acc_r6 * 100:.2f}%")
        print_top_features(rf_r6, X_r6.columns)
        joblib.dump(rf_r6, os.path.join(output_dir, "rf_run6_chronological_notime.joblib"))
        print("="*70)
    else:
        print("\nSkipping Runs 5 & 6: No primary timestamp column found for chronological sorting.")

if __name__ == "__main__":
    main()
