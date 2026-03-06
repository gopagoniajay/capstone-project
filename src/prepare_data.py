from preprocessing import preprocess_signals, create_gpu_tensors
from data_loader import load_wesad_subject
import numpy as np
import os

# Define subjects to process (All subjects in WESAD)
# S12 is missing in WESAD usually, checking available folders
wesad_path = 'data/WESAD'
subjects = [s for s in os.listdir(wesad_path) if s.startswith('S') and os.path.isdir(os.path.join(wesad_path, s))]
print(f"Found subjects: {subjects}")

all_X = []
all_y = []
all_subjects = []

for subject in subjects:
    print(f"Processing {subject}...")
    subject_path = os.path.join(wesad_path, subject, f"{subject}.pkl")
    
    if not os.path.exists(subject_path):
        print(f"Warning: Data file not found for {subject} at {subject_path}, skipping.")
        continue

    try:
        raw_data = load_wesad_subject(subject_path)
        features, labels = preprocess_signals(raw_data)
        
        # Use subject ID string or convert to int if preferred. Using string for clarity in LOSO.
        # But for numpy array it's better to be consistent. Let's keep it as string or encode it later.
        # Ideally, we want an array of strings.
        X, y, sub = create_gpu_tensors(features, labels, subject_id=subject)
        
        all_X.append(X)
        all_y.append(y)
        all_subjects.append(sub)
        print(f"  -> {subject}: X={X.shape}, y={y.shape}")
    except Exception as e:
        print(f"Error processing {subject}: {e}")

print("Combining all data...")
if all_X:
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)
    final_subjects = np.concatenate(all_subjects, axis=0)
    
    print(f"Success! Total processed data shape: X={final_X.shape}, y={final_y.shape}")

    # Ensure output directory exists
    os.makedirs('data', exist_ok=True)

    # Save this for the models
    np.save('data/processed_X.npy', final_X)
    np.save('data/processed_y.npy', final_y)
    np.save('data/processed_subjects.npy', final_subjects)
else:
    print("No data processed!")