import torch
import numpy as np
import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from src.personalize import load_data

def check_overlap(X_a, X_b, name_a="Set A", name_b="Set B"):
    """
    Checks if any sample in X_a exists in X_b.
    Returns True if overlap exists (LEAKAGE DETECTED), False otherwise.
    """
    print(f"  Checking overlap between {name_a} ({len(X_a)} samples) and {name_b} ({len(X_b)} samples)...")
    
    # Use byte representation for exact matching of numpy arrays
    set_a = set([bytes(x) for x in X_a])
    set_b = set([bytes(x) for x in X_b])
    
    intersection = set_a.intersection(set_b)
    count = len(intersection)
    
    if count > 0:
        print(f"  [CRITICAL FAIL] Found {count} overlapping samples!")
        return True
    else:
        print(f"  [PASS] No overlap found.")
        return False

def verify_loso_integrity():
    print("\n=== Verifying Leave-One-Subject-Out (LOSO) Integrity ===")
    X, y, subjects = load_data()
    unique_subjects = np.unique(subjects)
    
    print(f"Found {len(unique_subjects)} subjects: {unique_subjects}")
    
    leakage_found = False
    
    # Simulate one fold
    test_subject = unique_subjects[0]
    print(f"Testing Fold: Subject {test_subject} as TEST")
    
    # Create masks
    test_mask = (subjects == test_subject)
    train_mask = ~test_mask
    
    # Get indices
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    # Check 1: ID Leakage
    # Are any indices shared? (This should be impossible by definition of logic, but good to sanity check)
    if len(set(train_indices).intersection(set(test_indices))) > 0:
        print("  [FAIL] Index overlap detected!")
        leakage_found = True
    else:
        print("  [PASS] Subject indices are disjoint.")
        
    # Check 2: Data Leakage (Content)
    # Could the same data point appear in another subject? (Unlikely for raw sensors, but possible if data processing is wrong)
    X_train = X[train_mask]
    X_test = X[test_mask]
    if check_overlap(X_train, X_test, "LOSO Train", "LOSO Test"):
        leakage_found = True
        
    if not leakage_found:
        print("-> LOSO Integrity Confirmed: CLEAN")
    else:
        print("-> LOSO Integrity FAILED")

def verify_personalization_integrity():
    print("\n=== Verifying Personalization Split Integrity (20/80) ===")
    X, y, subjects = load_data()
    unique_subjects = np.unique(subjects)
    
    # Pick a random subject
    target_subject = unique_subjects[0]
    mask = (subjects == target_subject)
    X_target = X[mask]
    y_target = y[mask]
    
    print(f"Subject {target_subject} Total Samples: {len(X_target)}")
    
    # Replicate the split logic from personalize.py
    # "train_size=0.2" -> 20% for Calibration (Train), 80% for Evaluation (Test)
    X_calib, X_eval, y_calib, y_eval = train_test_split(X_target, y_target, train_size=0.2, random_state=42, stratify=y_target)
    
    # Check 1: Split Ratios
    ratio_calib = len(X_calib) / len(X_target)
    ratio_eval = len(X_eval) / len(X_target)
    print(f"  Calibration Size: {len(X_calib)} ({ratio_calib*100:.2f}%)")
    print(f"  Evaluation Size:  {len(X_eval)} ({ratio_eval*100:.2f}%)")
    
    if 0.19 <= ratio_calib <= 0.21:
        print("  [PASS] Calibration split is approx 20%.")
    else:
        print(f"  [FAIL] Calibration split is {ratio_calib*100:.2f}%, expected ~20%.")
        
    # Check 2: Data Leakage
    check_overlap(X_calib, X_eval, "Calibration Set", "Evaluation Set")

if __name__ == "__main__":
    try:
        verify_loso_integrity()
        verify_personalization_integrity()
        print("\n[SUMMARY] All integrity checks completed.")
    except FileNotFoundError:
        print("\n[ERROR] Processed data not found. Please run src/prepare_data.py first.")
