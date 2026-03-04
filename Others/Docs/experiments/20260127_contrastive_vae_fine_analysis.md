# Contrastive+VAE（fine, 11-label）集約考察

最終更新: 2026-01-28

対象データ:
- per-run: `hoyo_v1_1/joint_training_results/comparisons/contrastive_vae_fine_seed3.md`
- agg: `hoyo_v1_1/joint_training_results/comparisons/contrastive_vae_fine_seed3_agg.md`
- 条件: Sarashina / fine / freeze=2000 → full=8000 / seeds=3 / temp・λ_vae・λ_cont sweep

---

## 1. 結論（ベスト設定）

**seed平均で最も高い設定**は以下。

- **λ_cont=2.0 / λ_vae=0.12 / temp=0.05**
- `avg_r@1 = 0.356 ± 0.224`（n=6）

> ただし分散が大きいので、**安定性重視なら次点候補**も併記してよい。

次点候補（平均値が僅差）:
- λ_cont=2.0 / λ_vae=0.06 / temp=0.02 → `0.346 ± 0.211`
- λ_cont=1.0 / λ_vae=0.06 / temp=0.05 → `0.334 ± 0.248`
- λ_cont=2.0 / λ_vae=0.06 / temp=0.03 → `0.328 ± 0.231`

**解釈**:
- 極端に高い λ_cont(3.0) や高 temp(0.1) は平均的に落ちる傾向。
- λ_vae を 0.12 に上げても平均が上がっている点は重要な示唆（VAE 正則化が効く）。

---

## 2. temp の影響（最重要）

temp sweep（λ_cont=2.0 / λ_vae=0.06）を見ると：

- temp=0.02 → `avg_r@1 = 0.346 ± 0.211`
- temp=0.03 → `0.328 ± 0.231`
- temp=0.07 → `0.316 ± 0.231`
- temp=0.10 → `0.214 ± 0.117`

**傾向**:
- temp が高すぎる（0.10）は明確に悪化。
- 0.02〜0.07 は大きな差が出ないが、**低温側が優位**。

**考察**:
- 小規模データ（train≈222）では、temp を下げてコントラストを強くした方が分離が出る。
- temp=0.05 付近は「極端すぎず安定」な落とし所。

---

## 3. λ_vae の影響

λ_vae sweep（temp=0.05 / λ_cont=2.0）：

- λ_vae=0.03 → `0.314 ± 0.174`
- λ_vae=0.06 → `0.302 ± 0.190`（n=18）
- λ_vae=0.12 → `0.356 ± 0.224`

**傾向**:
- **0.12 が最も高い平均**
- 0.03 は不安定、0.06 は中庸

**考察**:
- VAE 正則化がある程度強い方が、fine ラベルでの一般化に効いている可能性。
- ただし分散が大きいので、**最終比較では seed を増やす**のが望ましい。

---

## 4. λ_cont の影響

λ_cont sweep（temp=0.05 / λ_vae=0.06）：

- λ_cont=1.0 → `0.334 ± 0.248`
- λ_cont=2.0 → `0.302 ± 0.190`
- λ_cont=3.0 → `0.285 ± 0.202`

**傾向**:
- λ_cont=1.0 が平均で最も高いが、分散も大きい
- λ_cont を上げすぎると平均は低下

**考察**:
- 強い contrastive は分離を促すが、fine(11クラス)では過学習/不安定化しやすい。
- λ_cont=1.0〜2.0 が安全域。

---

## 5. 指標間の関係

- `t2m@1` は比較的高く出やすい。
- `m2t@1` が下がる条件（temp高 / λ_cont高）が多い。

**示唆**:
- **avg_r@1 と m2t@1 を主指標に置くのが堅い**。
- t2m は補助的（飽和寄り）として扱うのが安全。

---

## 6. 論文向けまとめ（短く書くなら）

> We observe that the temperature parameter has the largest impact on retrieval performance. Lower temperatures (0.02–0.05) consistently outperform higher values (0.1), indicating that stronger contrastive separation is beneficial in the low-data regime. Increasing the VAE loss weight to 0.12 further improves avg R@1, suggesting that stronger regularization helps fine-grained generalization. In contrast, excessively large contrastive weights (λ_cont=3.0) reduce average performance, implying over-separation and instability. Overall, the best-performing configuration is λ_cont=2.0, λ_vae=0.12, temp=0.05, though variance across seeds remains high; thus, we recommend reporting mean±std over multiple seeds.

---

## 7. 次のアクション（推奨）

1. **最終候補2つ**で seeds を増やす（5以上）
   - A: λ_cont=2.0 / λ_vae=0.12 / temp=0.05
   - B: λ_cont=1.0 / λ_vae=0.06 / temp=0.05
2. `avg_r@1` と `m2t@1` を主表にし、`t2m@1` は付録扱い
3. temp曲線（0.02→0.10）をメイン図に掲載

