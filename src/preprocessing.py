import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler

def preprocess_signals(data, target_fs=32):
    # Extract & Resample
    ecg = signal.resample(data['signal']['chest']['ECG'].flatten(), int(len(data['label']) * target_fs / 700))
    eda = signal.resample(data['signal']['wrist']['EDA'].flatten(), int(len(data['label']) * target_fs / 4))
    
    # Label Sync
    # Ensure all have the same length (trim to stored minimum)
    min_len = min(len(ecg), len(eda))
    ecg = ecg[:min_len]
    eda = eda[:min_len]

    indices = np.linspace(0, len(data['label'])-1, min_len).astype(int)
    labels = data['label'][indices]

    # Standardize (Crucial for Neural Networks)
    # Using RobustScaler to handle outliers better (Subject S6 issue)
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    features = np.stack([ecg, eda], axis=1)
    features = scaler.fit_transform(features)
    
    return features, labels

def create_gpu_tensors(features, labels, subject_id=None, window_size=320): # 10-second windows
    X, y, subjects = [], [], []
    for i in range(0, len(features) - window_size, window_size // 2):
        window = features[i:i + window_size]
        label = np.bincount(labels[i:i + window_size]).argmax()
        if label in [1, 2, 3]: # Baseline, Stress, Amusement
            X.append(window)
            y.append(label - 1)
            if subject_id is not None:
                subjects.append(subject_id)
            
    if subject_id is not None:
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(subjects)
    else:
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)