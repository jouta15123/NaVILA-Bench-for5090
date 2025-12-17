
import json
import pickle
import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

data_dir = "data"
json_files = glob.glob(os.path.join(data_dir, "*.json"))

X = []
y = []
labels_map = {}

print(f"Loading data from {len(json_files)} files...")

for json_path in json_files:
    file_id = os.path.basename(json_path).replace(".json", "")
    pickle_path = os.path.join(data_dir, f"{file_id}.pickle")
    
    if not os.path.exists(pickle_path):
        continue
        
    with open(json_path, 'r') as f:
        meta = json.load(f)
        label = meta.get('annotation', {}).get('instruction', 'unknown')
        if label == "unknown": continue

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f) # (T, 14, 2)
        if isinstance(data, np.ndarray) and len(data.shape) == 3:
            # Feature Engineering
            # 1. Total Length
            length = data.shape[0]
            
            # 2. Mean Velocity (Speed)
            if length > 1:
                vel_seq = np.linalg.norm(np.diff(data, axis=0), axis=2) # (T-1, 14)
                mean_vel = vel_seq.mean()
                std_vel = vel_seq.std()
                max_vel = vel_seq.max()
            else:
                mean_vel = 0
                std_vel = 0
                max_vel = 0
            
            # 3. Motion Magnitude (Spatial Variance / Energy)
            # data: (T, 14, 2)
            # Centered around 0 for the whole sequence to measure spread
            centered_data = data - data.mean(axis=(0, 1))
            spatial_std = np.linalg.norm(centered_data, axis=2).std()
            
            # 4. Joint Volume (Mean distance from center of mass per frame)
            # (T, 14, 2) -> mean(axis=1) -> (T, 2)
            com = data.mean(axis=1, keepdims=True)
            dist_from_com = np.linalg.norm(data - com, axis=2) # (T, 14)
            mean_spread = dist_from_com.mean()
            
            features = [length, mean_vel, std_vel, max_vel, spatial_std, mean_spread]
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Data shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Simple Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_scaled, y, cv=cv)

print(f"\n--- Separability Analysis (Baseline Random Forest) ---")
print(f"Features used: [Length, MeanVel, StdVel, MaxVel, SpatialStd, MeanSpread]")
print(f"Mean Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
print(f"Chance Level (1/11): {1/11:.4f}")

# Detailed report on full data train/test split (just to see confusion)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix (Row=True, Col=Pred) ---")
# Get unique labels from y to ensure order
unique_labels = sorted(list(set(y)))
cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
print(f"{'':<12} " + " ".join([f"{l[:4]:<4}" for l in unique_labels]))
for i, row in enumerate(cm):
    print(f"{unique_labels[i]:<12} " + " ".join([f"{val:<4}" for val in row]))

if scores.mean() > 0.5:
    print("\nCONCLUSION: YES, highly separable. Even simple stats give good accuracy.")
elif scores.mean() > 0.3:
    print("\nCONCLUSION: POSSIBLY. There is signal, but significant overlap in simple stats. Deep learning should do better.")
else:
    print("\nCONCLUSION: DIFFICULT. Very high overlap in statistics.")
