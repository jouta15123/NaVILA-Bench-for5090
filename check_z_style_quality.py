import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# joint_training_results から latent snapshot を読む
snapshot_path = Path('hoyo_v1_1/joint_training_results/latent_snapshot_final.npz')
if not snapshot_path.exists():
    print("latent_snapshot_final.npz がまだないよ。学習終わってからやな。")
    exit()

data = np.load(snapshot_path)
z_m = data['z_m']  # (N, 512) - モーション潜在
labels_idx = data['labels_idx']  # (N,) - ラベルインデックス
label_list = data['label_list']  # (11,) - ラベル名
z_s_cls = data['z_s_cls']  # (11, 512) - スタイル埋め込み

# coarse グループ定義
coarse_groups = {
    "速い系": ["すたすた", "せかせか", "てくてく"],
    "遅い系": ["とぼとぼ", "のろのろ"],
    "重い系": ["どっしどっし", "のしのし"],
    "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
}

# ラベル名からインデックスマップ
label_to_idx = {lab: i for i, lab in enumerate(label_list)}

# 各モーションの coarse ラベルを決める
coarse_labels = []
for idx in labels_idx:
    lab = label_list[idx]
    for coarse, fines in coarse_groups.items():
        if lab in fines:
            coarse_labels.append(coarse)
            break
    else:
        coarse_labels.append("通常")  # 通常は含めないけど念のため

# スタイル埋め込みの coarse 平均を計算
coarse_z_s = {}
for coarse in coarse_groups.keys():
    indices = [label_to_idx[lab] for lab in coarse_groups[coarse]]
    coarse_z_s[coarse] = z_s_cls[indices].mean(axis=0)

# 各モーション z_m と coarse z_s の類似度（cosine）を計算
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sims = []
for i, z in enumerate(z_m):
    coarse = coarse_labels[i]
    if coarse in coarse_z_s:
        sim = cosine_sim(z, coarse_z_s[coarse])
        sims.append((coarse, sim))

# 系統ごとの平均類似度を表示
sim_by_coarse = defaultdict(list)
for coarse, sim in sims:
    sim_by_coarse[coarse].append(sim)

print("学習後の z_m と z_style の類似度チェック:")
for coarse in coarse_groups.keys():
    if coarse in sim_by_coarse:
        mean_sim = np.mean(sim_by_coarse[coarse])
        print(f"{coarse}: 平均類似度 {mean_sim:.3f} (サンプル数 {len(sim_by_coarse[coarse])})")

# 簡単なヒストグラム
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, coarse in enumerate(coarse_groups.keys()):
    if coarse in sim_by_coarse:
        axes[i].hist(sim_by_coarse[coarse], bins=20, alpha=0.7, label=coarse)
        axes[i].set_title(f"{coarse} の z_m-z_style 類似度")
        axes[i].set_xlabel("Cosine Similarity")
        axes[i].set_ylabel("Frequency")
        axes[i].axvline(np.mean(sim_by_coarse[coarse]), color='red', linestyle='--', label=f'Mean: {np.mean(sim_by_coarse[coarse]):.3f}')
        axes[i].legend()

plt.tight_layout()
plt.savefig('hoyo_v1_1/joint_training_results/style_similarity_check.png', dpi=150)
plt.close()
print("類似度ヒストグラムを保存した: hoyo_v1_1/joint_training_results/style_similarity_check.png")




