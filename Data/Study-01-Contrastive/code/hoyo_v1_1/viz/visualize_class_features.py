
import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from math import pi

# --- Configuration ---
DATA_DIR = "hoyo_v1_1/data"
OUTPUT_DIR = "hoyo_v1_1/viz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set Japanese Font
possible_fonts = ['Noto Sans CJK JP', 'IPAexGothic', 'TakaoGothic', 'VL Gothic']
selected_font = None
for f in font_manager.fontManager.ttflist:
    if any(pf in f.name for pf in possible_fonts):
        selected_font = f.name
        break
    if 'CJK' in f.name: # Fallback to any CJK
        selected_font = f.name
        break

if selected_font:
    print(f"Using font: {selected_font}")
    plt.rcParams['font.family'] = selected_font
else:
    print("WARNING: No suitable Japanese font found. Labels may not display correctly.")

# --- Data Loading & Feature Extraction ---

def calculate_features(data):
    # data: (T, J, D)
    T, J, D = data.shape
    
    if T < 3: # Need at least 3 frames for acceleration
        return None

    # Speed ( Velocity )
    # (T-1, J, D) -> norm -> (T-1, J) -> mean
    velocities = np.linalg.norm(np.diff(data, axis=0), axis=2)
    mean_speed = np.mean(velocities)

    # Acceleration
    # (T-2, J, D) -> norm -> (T-2, J) -> mean
    accelerations = np.linalg.norm(np.diff(data, n=2, axis=0), axis=2)
    mean_accel = np.mean(accelerations)

    # Motion Extent (Range of Motion)
    # Range of each joint over time: (J, D) -> mean over D -> (J,) -> mean over J
    # Or just range of the whole body?
    # Let's use: Average Dynamic Range per Joint
    ranges = np.max(data, axis=0) - np.min(data, axis=0) # (J, D)
    motion_extent = np.mean(ranges) # Mean over joints and dims
    
    # Duration (Frames)
    duration = T

    return {
        "Speed": mean_speed,
        "Acceleration": mean_accel,
        "Motion Extent": motion_extent,
        "Duration": duration
    }

records = []
json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
print(f"Processing {len(json_files)} files...")

for json_path in json_files:
    file_id = os.path.basename(json_path).replace(".json", "")
    pickle_path = os.path.join(DATA_DIR, f"{file_id}.pickle")
    
    if not os.path.exists(pickle_path):
        continue
        
    try:
        with open(json_path, 'r') as f:
            meta = json.load(f)
            label = meta.get('annotation', {}).get('instruction', 'unknown')
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, np.ndarray) and len(data.shape) == 3:
            feats = calculate_features(data)
            if feats:
                feats['Label'] = label
                records.append(feats)
    except Exception as e:
        print(f"Error processing {file_id}: {e}")

df = pd.DataFrame(records)
print(f"Extracted features for {len(df)} samples.")
print("Unique labels:", df['Label'].unique())

# --- Visualization 1: Violin Plots ---

features = ["Speed", "Acceleration", "Motion Extent", "Duration"]
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.violinplot(x="Label", y=feature, data=df, ax=axes[i], inner="box", palette="muted")
    axes[i].set_title(f"Distribution of {feature} by Class", fontsize=14)
    axes[i].set_xlabel("")
    axes[i].set_ylabel(feature)
    axes[i].tick_params(axis='x', rotation=45, labelsize=10)

plt.tight_layout()
violin_out = os.path.join(OUTPUT_DIR, "feature_violin_plots.png")
plt.savefig(violin_out)
print(f"Saved violin plots to {violin_out}")

# --- Visualization 2: Radar Chart ---

# Normalize features to 0-1 range for Radar Chart
# We compare the MEAN of each class
df_mean = df.groupby('Label')[features].mean()

# Min-Max Normalization across the means to emphasize differences
# (Or should we normalize based on the global min/max of the raw data? 
#  Global min/max preserves the 'absolute' sense better, but min-max of means highlights relative diffs between classes best.)
# Let's use Global Min-Max of the MEANs to spread them out in the plot.
normalized_means = (df_mean - df_mean.min()) / (df_mean.max() - df_mean.min())

# Create Radar Chart
labels = list(df_mean.index)
num_vars = len(features)
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1] # Close the loop

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], features, color='grey', size=12)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=10)
plt.ylim(0, 1.1)  # Since we normalized 0-1, maybe allow a bit of headroom

# Plot each class
# Distinct colors
colors = sns.color_palette("husl", len(labels))

for i, label in enumerate(labels):
    values = normalized_means.loc[label].values.flatten().tolist()
    values += values[:1] # Close the loop
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=colors[i])
    ax.fill(angles, values, color=colors[i], alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Class Feature Profiles (Normalized Means)", size=16, y=1.1)

radar_out = os.path.join(OUTPUT_DIR, "feature_radar_chart.png")
plt.savefig(radar_out, bbox_inches='tight')
print(f"Saved radar chart to {radar_out}")

print("Done!")
