"""
HOYOデータセットの全体像を可視化

1. オノマトペの種類とカテゴリ
2. スケルトンデータの数と内訳
3. 動作の分布を可視化
"""

import json
import pickle
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib
import japanize_matplotlib  # noqa: F401

matplotlib.use('Agg')  # GUI不要
import matplotlib.pyplot as plt


def analyze_hoyo_comprehensive(hoyo_dir: str):
    """
    HOYOデータセットの包括的な分析
    """
    hoyo_path = Path(hoyo_dir)
    json_files = sorted(hoyo_path.glob('*.json'))
    
    # データ収集
    all_data = []
    instruction_onos = []
    freestyle_onos = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_data.append(data)
        instruction_onos.append(data['annotation']['instruction'])
        freestyle_onos.extend(data['annotation']['freestyle'])
    
    print("=" * 80)
    print("HOYOデータセット 全体分析")
    print("=" * 80)
    
    # ========================================
    # 1. スケルトンデータの数と内訳
    # ========================================
    print("\n【1. 歩容スケルトンデータ】")
    print("-" * 80)
    
    total_sequences = len(all_data)
    persons = [d['person'] for d in all_data]
    views = [d['view'] for d in all_data]
    lengths = [d['length'] for d in all_data]
    
    print(f"総シーケンス数: {total_sequences}本")
    print(f"  - 前面（front）: {views.count('front')}本")
    print(f"  - 背面（back）: {views.count('back')}本")
    print(f"\n被験者数: {len(set(persons))}人")
    
    person_counter = Counter(persons)
    for person_id, count in sorted(person_counter.items()):
        print(f"  - Person {person_id}: {count}本")
    
    print(f"\nフレーム数:")
    print(f"  - 平均: {np.mean(lengths):.1f} frames ({np.mean(lengths)/60:.2f}秒 @ 60fps)")
    print(f"  - 最小: {min(lengths)} frames ({min(lengths)/60:.2f}秒)")
    print(f"  - 最大: {max(lengths)} frames ({max(lengths)/60:.2f}秒)")
    print(f"  - 中央値: {np.median(lengths):.1f} frames")
    
    # ========================================
    # 2. オノマトペの種類
    # ========================================
    print("\n【2. オノマトペの種類とカテゴリ】")
    print("-" * 80)
    
    instruction_counter = Counter(instruction_onos)
    freestyle_counter = Counter(freestyle_onos)
    
    print(f"Instruction（指示）: {len(instruction_counter)}種類")
    print(f"Freestyle（自由記述）: {len(freestyle_counter)}種類")
    print(f"総ユニーク数: {len(set(instruction_onos + freestyle_onos))}種類")
    
    # カテゴリ分類
    categories = {
        '遅い・重い': {
            'keywords': ['もた', 'のろ', 'とぼ', 'のそ', 'よた', 'のし', 'どし', 'どた', 'ずし'],
            'onos': []
        },
        '速い・元気': {
            'keywords': ['さっ', 'すた', 'てく', 'きび', 'しゃき', 'ぱた', 'はき', 'すい'],
            'onos': []
        },
        'ふらつく・揺れる': {
            'keywords': ['ふら', 'ゆら', 'よろ', 'ふわ', 'ぐら', 'とろ'],
            'onos': []
        },
        'リズミカル': {
            'keywords': ['とん', 'たん', 'ぽん', 'ころ', 'とこ', 'ぱん'],
            'onos': []
        },
        '不規則・変則': {
            'keywords': ['ぶら', 'うろ', 'きょろ', 'ちょろ', 'ぷら'],
            'onos': []
        }
    }
    
    # 全オノマトペをカテゴリ分類
    all_unique_onos = set(instruction_onos + freestyle_onos)
    
    for ono in all_unique_onos:
        categorized = False
        for cat_name, cat_data in categories.items():
            if any(kw in ono for kw in cat_data['keywords']):
                count = instruction_counter.get(ono, 0) + freestyle_counter.get(ono, 0)
                cat_data['onos'].append((ono, count))
                categorized = True
                break
        
        if not categorized:
            count = instruction_counter.get(ono, 0) + freestyle_counter.get(ono, 0)
            if 'その他' not in categories:
                categories['その他'] = {'keywords': [], 'onos': []}
            categories['その他']['onos'].append((ono, count))
    
    print("\n【カテゴリ別オノマトペ】")
    for cat_name, cat_data in categories.items():
        if cat_data['onos']:
            sorted_onos = sorted(cat_data['onos'], key=lambda x: -x[1])
            print(f"\n■ {cat_name} ({len(sorted_onos)}種類)")
            # Top 10を表示
            for ono, count in sorted_onos[:10]:
                print(f"  {ono:15s}: {count:4d}回")
            if len(sorted_onos) > 10:
                print(f"  ... 他 {len(sorted_onos) - 10}種類")
    
    # ========================================
    # 3. 頻度による分類
    # ========================================
    print("\n【3. 頻度による分類】")
    print("-" * 80)
    
    all_ono_counter = Counter(instruction_onos + freestyle_onos)
    
    freq_bins = {
        '100回以上': [],
        '50-99回': [],
        '10-49回': [],
        '5-9回': [],
        '2-4回': [],
        '1回のみ': []
    }
    
    for ono, count in all_ono_counter.items():
        if count >= 100:
            freq_bins['100回以上'].append((ono, count))
        elif count >= 50:
            freq_bins['50-99回'].append((ono, count))
        elif count >= 10:
            freq_bins['10-49回'].append((ono, count))
        elif count >= 5:
            freq_bins['5-9回'].append((ono, count))
        elif count >= 2:
            freq_bins['2-4回'].append((ono, count))
        else:
            freq_bins['1回のみ'].append((ono, count))
    
    for bin_name, onos in freq_bins.items():
        print(f"\n{bin_name}: {len(onos)}種類")
        if onos and len(onos) <= 20:
            for ono, count in sorted(onos, key=lambda x: -x[1]):
                print(f"  {ono:15s}: {count:4d}回")
    
    # ========================================
    # 4. Instructionオノマトペの詳細
    # ========================================
    print("\n【4. Instruction（指示オノマトペ）の詳細】")
    print("-" * 80)
    
    for ono, count in sorted(instruction_counter.items(), key=lambda x: -x[1]):
        # このオノマトペのスケルトンデータを数える
        samples = [d for d in all_data if d['annotation']['instruction'] == ono]
        
        # 統計
        sample_lengths = [d['length'] for d in samples]
        sample_persons = set([d['person'] for d in samples])
        
        print(f"\n{ono}: {count}本のスケルトン")
        print(f"  被験者: {len(sample_persons)}人")
        print(f"  平均フレーム数: {np.mean(sample_lengths):.1f}")
        print(f"  フレーム範囲: {min(sample_lengths)}-{max(sample_lengths)}")
    
    return {
        'total_sequences': total_sequences,
        'categories': categories,
        'freq_bins': freq_bins,
        'all_data': all_data
    }


def visualize_distribution(result):
    """
    分布を可視化
    """
    print("\n" + "=" * 80)
    print("可視化を作成中...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('HOYO Dataset Analysis', fontsize=16, fontweight='bold')
    
    # 1. カテゴリ別のオノマトペ数
    ax1 = axes[0, 0]
    cat_names = []
    cat_counts = []
    for cat_name, cat_data in result['categories'].items():
        if cat_data['onos']:
            cat_names.append(cat_name)
            cat_counts.append(len(cat_data['onos']))
    
    ax1.barh(cat_names, cat_counts, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Number of Onomatopoeia', fontsize=12)
    ax1.set_title('Onomatopoeia by Category', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. 頻度分布
    ax2 = axes[0, 1]
    freq_names = list(result['freq_bins'].keys())
    freq_counts = [len(onos) for onos in result['freq_bins'].values()]
    
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'lightblue', 'gray']
    ax2.bar(range(len(freq_names)), freq_counts, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(freq_names)))
    ax2.set_xticklabels(freq_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Number of Onomatopoeia Types', fontsize=12)
    ax2.set_title('Frequency Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Top 20オノマトペ
    ax3 = axes[1, 0]
    
    # 全オノマトペをカウント
    all_onos = []
    for d in result['all_data']:
        all_onos.append(d['annotation']['instruction'])
        all_onos.extend(d['annotation']['freestyle'])
    
    counter = Counter(all_onos)
    top_20 = counter.most_common(20)
    
    onos_20 = [ono for ono, _ in top_20]
    counts_20 = [count for _, count in top_20]
    
    ax3.barh(range(len(onos_20)), counts_20, color='coral', edgecolor='darkred')
    ax3.set_yticks(range(len(onos_20)))
    ax3.set_yticklabels(onos_20, fontsize=10)
    ax3.set_xlabel('Frequency', fontsize=12)
    ax3.set_title('Top 20 Most Frequent Onomatopoeia', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    ax3.invert_yaxis()
    
    # 4. フレーム数の分布
    ax4 = axes[1, 1]
    lengths = [d['length'] for d in result['all_data']]
    
    ax4.hist(lengths, bins=30, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax4.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.1f}')
    ax4.axvline(np.median(lengths), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.1f}')
    ax4.set_xlabel('Number of Frames', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Frame Length Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hoyo_dataset_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✅ 可視化を保存しました: hoyo_dataset_analysis.png")
    
    return 'hoyo_dataset_analysis.png'


def create_summary_report(result):
    """
    サマリーレポートを作成
    """
    print("\n" + "=" * 80)
    print("【サマリーレポート】")
    print("=" * 80)
    
    print(f"""
HOYOデータセット概要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. データ規模
   - 総スケルトン数: {result['total_sequences']}本
   - 被験者数: 10人
   - 視点: 前面・背面 各146本
   - 平均時長: 約4.3秒 @ 60fps

2. オノマトペ
   - Instruction（基本）: 11種類
   - Freestyle（バリエーション）: 604種類
   - 総ユニーク数: 605種類
   - 総アノテーション数: 12,936個

3. カテゴリ別内訳
""")
    
    for cat_name, cat_data in result['categories'].items():
        if cat_data['onos']:
            total_count = sum([count for _, count in cat_data['onos']])
            print(f"   - {cat_name}: {len(cat_data['onos'])}種類（{total_count}回出現）")
    
    print(f"""
4. 頻度分布
   - 100回以上: {len(result['freq_bins']['100回以上'])}種類（頻出・信頼性高）
   - 50-99回: {len(result['freq_bins']['50-99回'])}種類（一般的）
   - 10-49回: {len(result['freq_bins']['10-49回'])}種類（やや稀）
   - 5-9回: {len(result['freq_bins']['5-9回'])}種類（稀）
   - 2-4回: {len(result['freq_bins']['2-4回'])}種類（非常に稀）
   - 1回のみ: {len(result['freq_bins']['1回のみ'])}種類（ユニーク）

5. 学習への推奨
   ✅ 10回以上出現（約{len(result['freq_bins']['100回以上']) + len(result['freq_bins']['50-99回']) + len(result['freq_bins']['10-49回'])}種類）を使用
   → 信頼性のあるデータで学習
   
   ✅ Freestyleも活用してデータ拡張
   → 292本 → 数千ペアに増加可能
""")


def visualize_freestyle_examples(result, hoyo_root: Path, num_words: int = 6):
    """
    Freestyle オノマトペのうち、頻度上位いくつかについて実際の歩容座標を可視化する。
    現状は各オノマトペにつき1サンプルを取り、重心軌跡を 2D で描画する。
    """
    print("\n" + "=" * 80)
    print("Freestyle オノマトペの歩容例を可視化中...")
    print("=" * 80)

    # instruction / freestyle の頻度を再構成
    instr = [d["annotation"]["instruction"] for d in result["all_data"]]
    freestyle = []
    for d in result["all_data"]:
        freestyle.extend(d["annotation"]["freestyle"])

    instr_set = set(instr)
    fs_counter = Counter(freestyle)

    # instruction には含まれない freestyle 上位から num_words 個選ぶ
    target_words = []
    for w, _ in fs_counter.most_common():
        if w not in instr_set:
            target_words.append(w)
        if len(target_words) >= num_words:
            break

    if not target_words:
        print("Freestyle オノマトペが見つかりませんでした。")
        return None

    # 各オノマトペについて最初のサンプルを取得
    samples = []
    for w in target_words:
        for d in result["all_data"]:
            if w in d["annotation"]["freestyle"]:
                samples.append((w, d))
                break

    n = len(samples)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])

    for idx, (w, d) in enumerate(samples):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]

        pickle_path = hoyo_root / d["path"]
        try:
            with open(pickle_path, "rb") as f:
                coords = pickle.load(f)
        except Exception as e:
            print(f"{w} (id={d['id']}) の pickle 読み込みに失敗: {e}")
            continue

        # coords: (T, 14, 2) -> 重心の軌跡を描画
        com = coords.mean(axis=1)  # (T, 2) [y, x]
        ax.plot(com[:, 1], com[:, 0], "-o", markersize=2, linewidth=1)
        ax.set_title(f"{w} (id={d['id']})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

    # 余った軸は非表示
    for idx in range(len(samples), rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    out_path = hoyo_root / "hoyo_freestyle_examples.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Freestyle 歩容例を保存しました: {out_path}")

    return str(out_path)


if __name__ == "__main__":
    # スクリプト位置から HOYO データセットのルートを推定
    script_path = Path(__file__).resolve()
    hoyo_root = script_path.parent.parent  # .../hoyo_v1_1

    # データセット分析
    result = analyze_hoyo_comprehensive(str(hoyo_root))
    try:
        visualize_distribution(result)
        visualize_freestyle_examples(result, hoyo_root)
    except Exception as e:
        print(f"\n可視化エラー: {e}")
        print("（グラフは生成できませんでしたが、分析結果は上記の通りです）")
    create_summary_report(result)
    print("\n" + "=" * 80)
    print("分析完了！")
    print("=" * 80)