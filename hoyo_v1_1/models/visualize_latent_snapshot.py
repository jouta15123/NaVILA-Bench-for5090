import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


def pca_2d(x: np.ndarray):
    """
    Simple PCA to 2D using SVD (no external dependencies).
    Returns projected points, mean vector, and top-2 principal directions.
    """
    x_mean = x.mean(axis=0, keepdims=True)
    x_center = x - x_mean
    # x_center = U S Vt, rows of Vt are principal directions
    _, _, Vt = np.linalg.svd(x_center, full_matrices=False)
    components = Vt[:2]  # (2, D)
    x_2d = x_center @ components.T  # (N, 2)
    return x_2d, x_mean, components


def main():
    parser = argparse.ArgumentParser(description="Visualize motion latents (PCA 2D scatter).")
    parser.add_argument(
        "--snapshot",
        type=str,
        default="hoyo_v1_1/joint_training_results/latent_snapshot_final.npz",
        help="Path to latent snapshot .npz file produced by train_motionclip_joint.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="latent_pca.png",
        help="Output PNG path for the visualization.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        choices=["fine", "coarse"],
        default="fine",
        help=(
            "How to display labels:\n"
            "- 'fine': use snapshot labels as-is (11オノマトペ or 4スタイル).\n"
            "- 'coarse': 11オノマトペを 4 スタイル群（速い系/遅い系/重い系/ふらふら系）にまとめて表示。"
        ),
    )
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot file not found: {snapshot_path}")

    data = np.load(snapshot_path, allow_pickle=True)
    z_m = data["z_m"]  # (N, D)
    labels_idx = data["labels_idx"]  # (N,)
    splits = data["splits"]  # (N,)
    label_list = data["label_list"]  # (L,)
    z_s_cls = data["z_s_cls"]  # (L, D)

    # ラベル文字列に揃えておく
    label_list = [str(l) for l in label_list]

    # --- ラベルモードに応じて、表示用ラベル / プロトタイプを準備 ---
    # デフォルトは snapshot の粒度そのまま（fine / coarse どちらでも）
    plot_label_names = label_list
    plot_labels_idx = labels_idx
    plot_z_s = z_s_cls

    if args.label_mode == "coarse":
        # HOYO の coarse スタイル定義（train_motionclip_joint.py / visualize_coarse_analysis.py と同じ）
        COARSE_GROUPS = {
            "速い系": ["すたすた", "せかせか", "てくてく"],
            "遅い系": ["とぼとぼ", "のろのろ"],
            "重い系": ["どっしどっし", "のしのし"],
            "ふらふら系": ["ぶらぶら", "よたよた", "よろよろ"],
        }

        # もし snapshot 自体がすでに coarse ラベル（4つ）なら、そのまま使う。
        # 11 オノマトペが入っている場合は、それらを 4 群にマージする。
        label_to_idx = {lab: i for i, lab in enumerate(label_list)}

        # 1) semantic prototypes を 4 群にまとめる
        coarse_names = []
        coarse_protos = []
        for coarse_lab, fine_labs in COARSE_GROUPS.items():
            idxs = [label_to_idx[fl] for fl in fine_labs if fl in label_to_idx]
            if not idxs:
                continue
            coarse_names.append(coarse_lab)
            coarse_protos.append(z_s_cls[idxs].mean(axis=0))
        if coarse_protos:
            plot_z_s = np.stack(coarse_protos, axis=0)
            plot_label_names = coarse_names

        # 2) 各サンプルのラベルを coarse インデックスに変換
        coarse_name_to_idx = {name: i for i, name in enumerate(plot_label_names)}
        new_labels_idx = []
        for li in labels_idx:
            lab = label_list[li]
            # すでに coarse 名ならそのまま
            if lab in coarse_name_to_idx:
                new_labels_idx.append(coarse_name_to_idx[lab])
                continue
            # オノマトペを coarse グループにマッピング
            mapped = None
            for coarse_lab, fine_labs in COARSE_GROUPS.items():
                if lab in fine_labs and coarse_lab in coarse_name_to_idx:
                    mapped = coarse_name_to_idx[coarse_lab]
                    break
            # 「通常」などマップできないものは -1 として無視する
            if mapped is None:
                new_labels_idx.append(-1)
            else:
                new_labels_idx.append(mapped)
        plot_labels_idx = np.asarray(new_labels_idx, dtype=int)

    # PCA to 2D on motion latents
    z_2d, mean_vec, components = pca_2d(z_m)
    # Project semantic prototypes into the same 2D space
    z_s_center = plot_z_s - mean_vec
    z_s_2d = z_s_center @ components.T  # (L_plot, 2)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for visualization. "
            "Please install it with `pip install matplotlib`."
        ) from exc

    plt.figure(figsize=(8, 6))

    # Plot motion latents per split / label
    markers = {"train": "o", "test": "x"}
    for split_name in sorted(set(splits)):
        split_mask = splits == split_name
        for lab_idx, lab in enumerate(plot_label_names):
            # ラベルが -1（coarse にマップできない）なサンプルはスキップ
            mask = split_mask & (plot_labels_idx == lab_idx)
            if not np.any(mask):
                continue
            plt.scatter(
                z_2d[mask, 0],
                z_2d[mask, 1],
                s=10,
                alpha=0.6,
                marker=markers.get(split_name, "."),
                label=f"{lab} ({split_name})",
            )

    # Plot semantic prototypes as large stars
    plt.scatter(
        z_s_2d[:, 0],
        z_s_2d[:, 1],
        s=120,
        marker="*",
        edgecolors="k",
        facecolors="none",
        linewidths=1.0,
        label="sem prototypes",
    )

    plt.title("Motion / Semantic Latent Space (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", fontsize=8)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved latent visualization to {out_path}")


if __name__ == "__main__":
    main()


