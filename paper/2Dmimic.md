はい、承知いたしました。提供された論文の全文を、Obsidianでの数式記述形式に合わせて日本語に翻訳します。

---

**物理シミュレーションされた3Dキャラクターの2Dモーション生成と模倣による制御学習**

Jianan Li¹, Xiao Chen¹, Tao Huang²,³ Tien-Tsin Wong⁴
¹ 香港中文大学 ² 上海AI研究所 ³ 上海交通大学 ⁴ モナシュ大学

**図1.** 提案手法Mimic2DMは、ワイルドな動画から抽出された2Dモーションシーケンスを直接模倣することで、ダイナミックな人間のダンス、複雑なボールインタラクション、機敏な動物の動きを含む、多様なモーションタイプのキャラクターコントローラーを効果的に学習します。

**概要**

ビデオデータは、3Dキャラクターモーションコントローラー学習のためのモーションキャプチャデータよりもコスト効率が高いですが、ビデオから直接、リアルで多様な振る舞いを合成することは依然として困難です。以前のアプローチは、物理ベースの模倣のために3D軌道を取得するために、既存のモーション再構成技術に依存することが一般的でした。これらの再構成手法は、一般化性に課題があり、3Dトレーニングデータ（希少な場合がある）を必要とするか、物理的に妥当なポーズを生成できないため、人間とオブジェクトのインタラクション（HOI）や人間以外のキャラクターのような困難なシナリオへの応用を妨げています。本稿では、Mimic2DMを紹介します。これは、ビデオから抽出された広く利用可能な2Dキーポイント軌道のみから直接、そして唯一のデータソースとして、制御ポリシーを学習する新しいモーション模倣フレームワークです。再投影誤差を最小化することで、単一視点からの任意の2D参照モーションを物理シミュレーションで追跡できる、汎用的な単一視点2Dモーション追跡ポリシーを学習します。このポリシーは、異なる、あるいはわずかに異なる視点からキャプチャされた多様な2Dモーションでトレーニングされると、複数のビューを集約することによって3Dモーション追跡能力を獲得できます。さらに、トランスフォーマーベースの自己回帰型2Dモーションジェネレーターを開発し、階層的制御フレームワークに統合しました。ここで、ジェネレーターは高品質な2D参照軌道を生成し、追跡ポリシーをガイドします。提案手法は多用途であり、明示的な3Dモーションデータに依存することなく、ダンス、サッカーのドリブル、動物の動きなど、さまざまなドメインで物理的に妥当で多様なモーションを効果的に合成できることを示します。プロジェクトウェブサイト: https://jiann-li.github.io/mimic2dm/

**1. はじめに**

物理的にシミュレートされたキャラクターにリアルなモーションや妥当なオブジェクトインタラクションを実行させることは、コンピュータアニメーションやロボット工学における根本的かつ困難な問題であり続けています。最近では、モーション模倣技術がモーションキャプチャ（MoCap）データを活用して物理ベースのキャラクターコントローラーをトレーニングし、シミュレートされた仮想キャラクター上で非常にダイナミックで物理的にリアルなモーションを生成する上で目覚ましい成果を上げています [7, 12, 13, 29, 35, 52]。

しかし、高品質な3D MoCapデータの収集は、多数の熟練したパフォーマーと特殊なキャプチャシステムを必要とするため、コストが高く、労働集約的です。

高品質な3D MoCapデータの希少性に対処するため、最近の研究では、代替データソースとしてビデオの活用が検討されています。ほとんどの既存手法 [24, 30, 59, 62] は、学習のための物理ベーススキルをシミュレートされた3Dモーションを推定するために、既存の人間モーション再構成技術を利用しています。高度なトレーニングベースの推定手法は、人間モーションの再構成において顕著な精度とリアリズムを達成できますが、そのパフォーマンスは、トレーニングのための広範な高品質3Dデータに大きく依存しており、3Dデータが希少なドメイン（人間とオブジェクトのインタラクションや人間以外のモーションなど）での適用性を制限しています。さらに、これらの手法は、物理的制約の欠如により、物理的に妥当でないモーションをもたらすことが多く、これが後続のモーション模倣を妨げます。

ビデオから推定された信頼性の低い3Dモーションでのトレーニングとは対照的に、いくつかの研究では、ビデオ映像から抽出された2Dモーションを直接利用する可能性を示しており、さまざまな3Dタスクで成功を収めています [3, 11, 16, 33, 44]。この2Dデータは、オブジェクトインタラクションや人間以外の（動物の）動きを含む、幅広いスケルトンに対して容易にアクセスでき、ビデオから抽出できます。さらに、ビデオで検出された2Dキーポイントモーションは、映像に存在する元の動きを正確に反映する、偏りのない2D証拠を提供します。2Dデータを使用する際の主な課題は、深度情報の欠落です。2D事前知識と幾何学的制約の組み合わせは、視覚的に妥当な3Dポーズをもたらす可能性がありますが、結果として得られるモーションは物理的に制限されており、モーション模倣のための高品質データとして直接利用することはできません。

本論文では、Mimic2DMを紹介します。これは、ビデオから抽出された広く利用可能な2Dモーションデータのみに依存することで、人間とオブジェクトのインタラクション（HOI）や動物の歩行など、さまざまな複雑な物理ベーススキルを獲得できる汎用的な模倣学習フレームワークです。2Dモーションデータを活用するために、3D再構成と物理ベースモーション模倣を単一の再投影最小化タスクに統合した、物理ベースの2Dモーション追跡を定式化します。これは強化学習（RL）によって最適化されます。物理的制約を含めることにより、学習されたポリシーは、深度が欠落した2Dデータから直接模倣することで、物理的に正しい3Dモーションを合成できます。これに基づいて、ビューに依存しない追跡ポリシーを導入します。この設計は、ポリシーがデータ内の多様な視点から利益を得られるようにし、よりリアルな3Dモーションを学習できるようにするだけでなく、汎用的な3Dモーション追跡タスクのためのマルチビュー追跡ポリシーへの簡単な拡張を容易にします。単一視点モーション追跡のトレーニング効率を向上させるために、適応的な状態初期化戦略と、再投影誤差ベースの早期終了基準を提案します。最後に、このフレームワークを新しいモーション合成などの生成タスクに拡張するために、追跡ポリシーと2Dモーションジェネレーターを統合した階層的制御構造を採用します。ここで、2Dモーションはモーションジェネレーターと制御ポリシー間のインターフェースとして機能します。

サッカーボールとの熟練したインタラクションや、ロボット犬の非常にダイナミックな動きなど、さまざまな困難な物理スキルを、ワイルドなビデオから抽出された2Dモーションシーケンスのみを使用して学習できることを実証します。また、提案されたビューに依存しない2D追跡ポリシーが、普遍的な追跡能力を示すことも実証します。カジュアルなビデオはさまざまな視点から撮影される可能性が高いため、このデータでトレーニングされた単一視点ポリシーは、ビュー集約メカニズムを介してマルチビュー追跡ポリシーに効果的に拡張でき、3D追跡を実行できるようになります。決定的に、2Dデータのみでトレーニングされた場合でも、提案手法が3Dモーションでトレーニングされた従来のメソッドと同等の3D追跡精度を達成することを示します。さらに、2Dモーションジェネレーターとの階層的フレームワークに拡張して、モーション合成と条件付き制御を行うことで、提案手法の生成能力を強調します。このセットアップでは、提案された自己回帰型2Dモーションジェネレーターが、追跡ポリシーを効果的にガイドするために必要な高品質な2Dモーションシーケンスの生成において、現在の拡散ベースモデルを上回ることを示します。

**2. 関連研究**

**2.1. 物理ベースキャラクター制御**
リアルで物理的に妥当なキャラクターの振る舞いを達成することは、コンピュータアニメーションにおける主要な目標および課題です。この目的のために、仮想キャラクターの複雑なモーションダイナミクスと衝突インタラクションをエミュレートするために物理シミュレーションが使用されています。初期の研究では、物理ベースのキャラクターアニメーションは、従来の最適化ベースの制御戦略といくつかのヒューリスティックなルールを組み合わせて、主に運動能力に焦点を当てていました [4, 5, 8, 34, 57, 58]。その後、強化学習（RL）が導入され、シミュレートされたキャラクターが基本的な運動能力 [27, 28, 36, 38] から熟練したスポーツスキル [2, 18] まで、幅広い複雑なスキルを習得できるようになりました。しかし、効果的な報酬関数の設計には通常、専門知識が必要であり、強化学習コントローラーによって生成される振る舞いは、しばしば不規則なモーションパターンを示します。これらの問題に対処するために、MoCapデータは物理シミュレートされたキャラクターコントローラーのトレーニングに利用されています。これは、明示的なモーション追跡報酬 [1, 14, 19, 26, 29] または識別子から導出される暗黙のモーションスタイル報酬 [7, 31, 49] を使用して達成できます。これらのアプローチにより、より自然で一貫性のあるキャラクターの振る舞いを学習できます。

学習されたスキルを幅広い下流タスクに再利用するために、最近の研究では、VAE [20, 47, 54, 55] やGAN [6, 9, 10, 32, 39] のような潜在ベースの生成モデルを探索し、モーションクリップを低次元の潜在空間にマッピングすることによって、再利用可能なモーションプリミティブを学習しています。これらのアプローチは、各下流タスクの事前トレーニングされた潜在表現スキルを制御する個別の高レベルポリシーを効率的に学習できます。別の研究ラインは、ユニバーサルモーション追跡コントローラーと運動学モーション生成モデル [15, 40, 51] の組み合わせに焦点を当てています。この階層的制御フレームワークは、多用途で物理的に妥当なモーション合成と制御 [37, 41, 48, 52] もサポートしています。しかし、これらのアプローチはすべて、トレーニングのために高品質な3D MoCapデータに依存しており、その適用性とスケーラビリティを大幅に制限しています。対照的に、私たちの手法はモーション模倣アプローチであり、2Dモーションデータのみを必要とするため、よりアクセスしやすく多用途です。

**2.2. ビデオからの物理スキルの学習**
モーションキャプチャ（MoCap）データと比較して、ビデオは物理ベーススキルを学習するためのよりアクセスしやすいソースです。Vondrakらによる初期の研究 [43] は、単眼ビデオからのシルエット損失を最小化することにより、物理シミュレーション環境でジャンプや体操の動きを再現しようとしました。コンピュータビジョンの最近の進歩により、ビデオからの3D人間ポーズの再構成が可能になり、これは物理ベースのキャラクターコントローラーのトレーニングに利用できます。初期の試みは、単一ビデオから導出された物理シミュレーターで個々のモーションインスタンスを再現することに焦点を当てていました。たとえば、Pengら [30] は、ビデオから推定された3Dポーズを追跡するモーション模倣パイプラインを提案しました。これに基づいて、Yuら [59] は、2D/3Dポーズや足の接触などの追加のヒントをポリシー学習に組み込むことで技術を進化させ、動的なカメラ移動を伴う長いビデオシーケンスから機敏なモーションを合成できるようにしました。新しいビデオクリップのために時間のかかる物理ベースのモーション模倣を回避するために、Yuanら [61] はリアルタイムの物理モーション推定アプローチSimPoeを導入しました。SimPoeは、モーションコレクターとして大規模な3DモーションデータセットAMASS [21] でトレーニングされたユニバーサル物理ベース追跡コントローラーを使用します。最近では、ビデオデータが、研究室環境 [24, 45, 46, 50, 62] でキャプチャするのが困難または高価な複雑なスキルを学習するための、優れたアクセス性とスケーラビリティを示しています。しかし、主な課題は、ビデオからの3Dポーズが物理的に信頼できないことです。したがって、このデータは、物理ベースの制御ポリシーをトレーニングするために使用される前に、広範な後処理、あるいは手動修正を必要とします。対照的に、私たちの手法は2Dポーズシーケンスから直接エンドツーエンド学習を行うため、「ワイルドな」ビデオに適用可能で、さまざまなキャラクターのスケルトンに適応できます。

**3. 方法**

**3.1. 模倣としての再投影最小化**
2Dモーションシーケンスを、座標のシリーズ $X \in \mathbb{R}^{T \times J \times 2}$で表されるものとします。ここで、$T$はモーションの長さ、$J$はキーポイントの数（スケルトンジョイントまたはオブジェクトランドマークを含む）です。私たちの目標は、ポリシー $\pi$を学習することです。このポリシーは、シミュレートされたキャラクターに物理的に妥当な3Dモーションを実行させるように制御し、その2D投影が与えられたカメラビューで、提供された2Dモーション参照に正確に一致するようにします。既存のアプローチは、2D証拠から3Dモーションを別々に再構成し、再構成された3Dモーションを模倣する制御ポリシーを学習することがよくあります。2Dから3Dへの逆変換の不良設定のため、再構成された3Dモーションは、特にオブジェクトインタラクションや人間以外の運動などの3D事前情報がないドメインでは、物理的に実行不可能であることがよくあります。この欠陥のある3D監督は、モーション模倣の大きな障害となり、しばしば学習の失敗につながります。これを解決するために、モーション再構成とモーション模倣を、以下の式で定義される物理ベースの2Dモーション追跡問題として統合することを提案します。

$$
\min_{\pi} \underset{s_0 \sim d(s_0)}{\mathbb{E}} \|P_{\pi}(C) - X\|_{s.t. f_{\pi} = 0} \quad (1)
$$

ここで、$P_{\pi}(C)$は、カメラビュー$C$でのポリシー $\pi$によって合成された3Dジョイント位置の2D投影、$d(s_0)$は初期状態の分布、$f_{\pi} = 0$はシミュレートされたキャラクターの物理的制約と基盤となるMDPダイナミクスを表します。統合された定式化により、エンドツーエンドのエラー最小化が可能になり、物理的制約を利用して結果の3Dモーションを正規化し、それによって物理的妥当性を保証します。

**3.2. ビューに依存しない2D追跡ポリシー**
再投影目的のみの最適化は、本質的に深度の曖昧さによって制限され、しばしば妥当でない、または不自然なポーズにつながります。この問題に対処するために、カジュアルなビデオは通常、さまざまな視点からキャプチャされ、全体としてモーションを描写するのに十分な情報を提供することに気づきました。この洞察に基づいて、任意のカメラビューで参照モーションの再投影誤差を最小化するように設計された、汎用的な2D追跡ポリシーをトレーニングすることを提案します。このようなポリシーは、データに存在するさまざまな視点からの2D再投影制約を満たすことを学習するため、暗黙的にモーションの3D理解を獲得します。さらに、このクロスビュー汎化機能により、ポリシーは特徴集約を介してマルチビューモーション追跡に効果的に拡張でき、3D監督なしで堅牢な3Dモーション追跡パフォーマンスを達成できます。

2D追跡ポリシーはニューラルネットワークとして表されます。このネットワークは、キャラクターの自己受容性 $s_{prop}^t$ と将来の2Dモーション参照 $o_{2D}^t$ の観測を入力として受け取り、アクション空間における対角ガウス分布としてシミュレートされたキャラクターのPD（比例・微分）ターゲットを予測します。

**ビューに依存しない2D観測**
任意のビューの2Dモーション追跡の汎化能力を向上させるために、観測から明示的なカメラビュー情報は意図的に省略し、ポリシーがシミュレーション内のキャラクターの実際の2D投影からのみ2Dモーション参照のビューポイントを推測するように強制します。提案された2D観測は次のように表されます。

$o_{2D|C}^t = [P(x_{3D}^t, C), x_{t:t+L}] \quad (2)$

ここで、$x_{t:t+L} \in \mathbb{R}^{L \times J \times 2}$は、$L$フレーム先読みする将来の2D参照モーションのクリップであり、$P(x_{3D}^t, C) \in \mathbb{R}^{J \times 2}$はシミュレーションにおける3Dキーポイント $x_{3D}^t \in \mathbb{R}^{J \times 3}$の2D投影を表します。

**3Dモーション追跡のためのビュー集約**
多様なビューデータは、よく制約されたモーション空間を提供し、それによってビューに依存しない追跡ポリシーによって合成される、より物理的に妥当な3Dモーションにつながります。しかし、単一の2D投影の本質的な曖昧さは依然として残っており、3D空間におけるポリシーの微細な制御可能性を制限します。図3に示すように、この問題を軽減するためにビュー集約技術を導入することにより、ポリシーをマルチビュー追跡ポリシーに適合させることを提案します。まず、汎用的な2D単一視点追跡ポリシーの機能を拡張して3Dモーション追跡を達成する方法を、単一視点目的をマルチビュー設定に再構築することによって実証します。K個のマルチビュー2Dモーションとその対応するカメラビュー $\{X_k, C_k\}_{k=1}^K$を考慮してマルチビュー追跡問題を定義します。これらはすべて単一の根本的な3Dモーションシーケンスから投影されます。すべてのビューにわたる総再投影誤差を最小化するポリシー $\pi$を見つける目的は次のとおりです。

$$
\min_{\pi} \underset{s_0 \sim d(s_0)}{\mathbb{E}} \sum_{k=1}^K \|P_{\pi}(C_k) - X_k\|_{s.t. f_{\pi} = 0} \quad (3)
$$

定義されたマルチビュー追跡問題の最適性が、汎用的な単一視点追跡ポリシーによって達成される最適性で十分であると仮定します。ポリシーが複数のビューからの2D観測 $\{o_{2D|C_1}^t, o_{2D|C_2}^t, \dots, o_{2D|C_K}^t\}$を受け入れられるようにするために、さまざまなビューからの情報を組み合わせるためのビュー集約戦略を考案しました。ニューラルネットワークの特徴空間で一般的に観察される線形性を利用して、すべてのビューからの2D観測の特徴埋め込みを平均化することを選択します。ビューに依存しない追跡コントローラーから適応されたマルチビューモーション追跡ポリシー $\pi_{mv}$は次のように定義されます。

$\pi_{mv}(a_t|s_{prop}^t, o_{2D|C_1}^t, \dots, o_{2D|C_K}^t) = \pi(a_t|s_{prop}^t, o_{agg}^t) \quad (4)$

ここで、集約された特徴 $o_{agg}^t$は、ビュー固有の特徴の平均を計算することによって計算されます。

$o_{agg}^t = \frac{1}{N} \sum_{i=1}^N \phi(o_{2Di}^t) \quad (5)$

ここで、$\phi(\cdot)$は単一視点2D観測の特徴抽出器を表します。提案されたビュー集約戦略は、ビューに依存しない追跡ポリシーに、ファインチューニングなしでマルチビューモーション追跡の機能を与え、それによって3D制御可能性を大幅に向上させます。

**3.3. 単一視点追跡のトレーニング**
ビューに依存しない2D追跡ポリシーのトレーニングは、3つの主要な適応を備えた3Dモーション追跡と同様の強化学習パラダイムに従います。再投影誤差を最小化するように特別に設計された2D追跡報酬、高品質な3D参照ポーズの欠如を解決するための適応的な状態初期化、および2Dモーション模倣トレーニングに合わせた早期終了戦略。

**2D追跡報酬**
式1の再投影誤差の最小化を奨励するために、エネルギー消費ペナルティによる正規化を伴う距離ベースの報酬関数 $r_t = w_p r_p^t + w_e r_e^t$ を使用します。ここで、$w_p$と$w_e$は報酬の重みです。2D追跡報酬 $r_p^t$は、投影された2Dキーポイントと参照2Dポイントとの間の不一致を測定することによって計算されます。

$r_p^t = \exp \left( -\alpha \sum_{j=1}^J \|P(x_{3D|j}^t) - x_j^t\| \right) \quad (6)$

ここで、$x_j^t$は時間 $t$ における $j$ 番目の参照キーポイントの座標、$P(x_{3D|j}^t)$はシミュレートされたキャラクタージョイントの投影された2D座標、$ \alpha$は正のスカラーです。エネルギー消費ペナルティ $r_e^t$は、次のように計算されます。

$r_e^t = -\sum_{j} \|\dot{q}_j^t \cdot \tau_j^t\|$

ここで、$\dot{q}_j^t$と$\tau_j^t$は、時間 $t$ におけるジョイント $j$ のジョイント速度とトルクを表します。

**適応的状態初期化**
初期状態の分布は、RLのトレーニング効率にとって重要です。従来の3Dモーション模倣では、広く採用されている参照状態初期化（RSI）[29]は、正確な3Dモーション状態に依存していますが、2Dデータのみが提供される場合は利用できません。不完全な3D再構成 [60, 61] または手動で指定されたデフォルトポーズを状態初期化に使用できますが、物理的に実行不可能な状態の発生は、ポリシー学習を大幅に妨げます。この制限に対処するために、クリティックネットワークを使用して推定された追跡パフォーマンスを評価することにより、最適でない初期状態を破棄することを提案します。状態の分布を予測するニューラルネットワークポリシーを学習する [30] アプローチに触発されて、代わりにデータバッファを使用して、トレーニングの安定性を向上させるための初期状態分布の離散表現を維持します。

具体的には、参照フレームごとに専用のバッファが維持され、初期は任意の初期3Dポーズで満たされます。トレーニング中、ポリシーロールアウトから取得された状態はクリティックネットワークを使用して評価され、高いクリティックスコアを達成した状態は、対応するフレームバッファに格納されます。優先探索を容易にするために、各格納された状態のサンプリング確率は、そのスコアに指数関数的に比例するように設定されます。

**再投影ベースの早期終了**
トレーニング効率を確保し、シミュレートされたキャラクターの回復不能な状態を回避するために、再投影誤差に基づいた早期終了メカニズムを導入します。具体的には、キャラクターの投影されたポーズが参照2Dポーズから大幅に逸脱すると、エピソードは終了します。

**3.4. 2Dインターフェースによる階層的制御**
トレーニングされたビューに依存しない2D追跡ポリシーは、キネマティック2Dモーション生成モデルとの統合により、生成タスクに拡張できます。これにより、階層コントローラーが形成されます。リアルタイムで無限長のモーション合成を可能にするために、トランスフォーマーベースの自己回帰型2Dモーションジェネレーターを提案します。このGPTライクなアーキテクチャは、高品質なモーション特性を維持しながら、オンザフライ生成を保証します。2Dモーションにおけるグローバルな動きを効率的にエンコードするために、2Dモーションの正規化された表現を導入し、2Dモーションジェネレーターの学習を容易にします。

**2Dモーションの正規化された表現**
生の2Dキーポイント座標は、横方向の変位（例：右から左への移動）や、カメラ軸に沿った移動によるスケールの変化（例：カメラに近づく、または遠ざかる）などの、グローバルなキャラクターの動きによる大きな変動を示すことがよくあります。この高い変動は、生成モデルのトレーニングを困難にします。この問題に対処するために、ニューラルネットワーク学習に対してより効率的でありながら、必要なグローバルモーション情報を保持する、正規化された2Dモーション表現を提案します。

より具体的には、各フレームについて、ルートの移動 $x_{root}$、スケール係数 $s$、およびローカルポーズ $\bar{x}$を、$x = (\bar{x}s) + x_{root}$の式で計算します。連続フレーム間の相対スケール変化は $\delta s_t = \log(s_t/s_{t-1})$ で与えられ、ルート移動の正規化されたシフトは $\delta x_{root}^t = (x_{root}^t - x_{root}^{t-1})/s_t$ と定義されます。相対2Dモーションシーケンスは、$x^{can} = [x^{can}_0, x^{can}_1, ..., x^{can}_T]$ として表され、ここで各 $x^{can}_t = (\bar{x}_t, \delta x_{root}^t, \delta s_t)$ です。最初のフレームのスケールとルート移動 $x_{root}^0$ と $s_0$ を与えると、絶対表現と相対表現の間で変換できます。順変換と逆変換は次のように定義されます。

$X^{can} = G(X), X = G^{-1}(X^{can}, x_{root}^0, s_0) \quad (7)$

**2Dモーション・トークナイザー**
冗長な情報を圧縮し、正規化された2Dモーションシーケンスをコンパクトなトークンのシーケンスに離散化するために、ベクトル量子化変分オートエンコーダー（VQ-VAE）[42]を使用します。長いシーケンスにわたる正確なグローバル移動とスケールの一貫性を維持するために、損失関数に共通の2D再構成項を追加します。これは、正規化された表現をグローバル座標空間に変換することによって計算されます。

$L_{rec} = \|X^{can} - \hat{X}^{can}\| + \omega \|G^{-1}(X^{can}) - G^{-1}(\hat{X}^{can})\|$

ここで、$ \omega$は共通の損失項の重み係数、$\hat{x}^{can}$はVQ-VAEによって再構成された共通の2Dモーションです。さらに、自己回帰型トークンデトークナイゼーションを実現するために、VQ-VAEアーキテクチャで因果的畳み込み層 [25] を採用しています。

**自己回帰型ジェネレーター**
VQ-VAEを使用して2Dモーションシーケンスを離散コードブックインデックス $c = [c_0, c_1, \dots, c_{T/l}]$ にトークン化することにより、データ分布は離散的で因数分解された分布としてモデル化されます。$p(c) = \prod_{i=1}^{T/l} p(c_i | c_0, \dots, c_{i-1})$ ここで $c_i \in \{1, \dots, K\}$ です。因果的トランスフォーマーを使用してこの分布をモデル化し、$p_{2D}(c_0, c_1, \dots, c_{T/l} | y)$ と表します。ここで $y$ はオプションの条件付け変数です（条件なし生成の場合は $y = \emptyset$）。モデルは、ターゲットトークンインデックスと予測確率との間のクロスエントロピー損失を最小化することによってトレーニングされます。

**階層的制御のインターフェース**
トレーニングされたビューに依存しない2D追跡ポリシーは、2Dモーションジェネレーターと統合して、さまざまな生成タスクに対応できる階層コントローラーを形成できます。生成された2Dモーション参照フレーム $\hat{x}_{t:t+L}^{can}$ は、まず $\hat{x}_{t:t+L} = G^{-1}(\hat{x}_{t:t+L}^{can})$ を介してグローバル表現に変換されます。生成されたグローバル座標の2Dモーションを、追跡ポリシーの参照モーションと見なします。ビューに依存しないトレーニングの恩恵を受けて、投影に使用されるビューポイントは任意に選択できます。この設計は、ビューポイントに対する剛性のある制約を排除し、コントローラーのさまざまなシナリオへの適用性を向上させます。

**4. 実験**

**4.1. 実験設定**
**データ**
フレームワークの有効性を、ワイルドなビデオと公開データセットの両方で評価します。HOIや人間以外のモーションなど、希少な3Dデータを持つ困難なシナリオからの学習能力を評価するために、サッカーのドリブルと動物の動きの2つの新しいデータセットをキュレーションしました。サッカーのドリブルは、複雑なサッカーのドリブルスキルを示すオンラインビデオから収集され、複雑なフットボールインタラクションダイナミクスが含まれています。ViTPose [53] を使用して、これらのビデオから2D人間ポーズとボールのバウンディングボックスを検出し、抽出しました。動物の動きもオンラインビデオからソースされ、歩行、走行、ジャンプなどのさまざまな犬の動きが含まれています。2Dキーポイントは、ドリブルデータセットと同じパイプラインで検出および処理されました。ベンチマークのために、AIST++データセット [17] も含めました。これは、複数の視点からキャプチャされたダイナミックなダンスビデオと、注釈付きの3D人間ポーズを含む大規模な公開データセットです。 [16] のプロトコルに従って、シーケンスごとに単一のカメラビューをランダムに選択することにより、単一視点2Dモーションデータを生成します。

**メトリクス**
定量的な評価には、次のメトリクスを採用します。成功率（Success Rate）は、制御ポリシーの堅牢性を測定し、すべての参照モーションに対する正常に追跡されたモーションの割合として定義されます。最大再投影誤差が100ピクセルを超える場合、トライアルは失敗と見なされます。2D追跡誤差（2D Tracking Error）は、2D空間におけるポリシーの追跡精度を評価し、キャラクターの投影ポーズと参照モーションとの間の平均再投影誤差として計算されます。2Dオブジェクト追跡誤差（2D Object Tracking Error）も同様に定義されますが、人間とオブジェクトのインタラクション（HOI）タスクにおけるオブジェクト追跡の精度を測定します。3D追跡誤差（3D Tracking Error）は、すべてのフレームにわたるジョイント位置誤差の平均をとることによって、3D参照モーションに対する追跡精度を評価します。ジッター（Jitters）は、ジョイント位置の3次導関数をモーションの滑らかさの尺度として計算することにより、合成モーションの物理的妥当性を定量化します。最後に、FIDは、合成モーションと参照モーションとの間の分布の違いを測定します。 [11] のプロトコルに従って、FIDスコアは合成ポーズの2D投影で計算されます。

**実装の詳細**
制御ポリシーは30Hzで実行され、60HzでダイナミクスをシミュレートするIsaac Gym [23] の物理シミュレーターと対話します。制御ポリシーは、512、256、256の隠れユニットを持つMLPネットワークです。ベクトル量子化変分オートエンコーダー（VQ-VAE）では、コードブックサイズを512、埋め込み次元を128に設定します。自己回帰型トランスフォーマーは、4つのレイヤー、4つのアテンションヘッド、128の埋め込み次元を持つデコーダーアーキテクチャを採用します。初期状態として初期ポーズを取得するために、再投影誤差を最小化するだけで最適化します。注：提案手法は初期状態に敏感ではなく、最先端のポーズ推定技術や手動指定など、さまざまな方法で取得できます。単一視点追跡ポリシーのトレーニングには、AIST++モーションとドリブルモーションの両方で、4つのNVIDIA P40 GPUを使用して約1週間かかります。動物のモーションの模倣は、同じリソースで約3日かかります。

**4.2. モーション模倣の評価**
**2段階アプローチとの比較**
困難なHOIおよび人間以外のモーションデータセット（サッカーのドリブルと動物）でアプローチを評価します。また、3Dポーズを推定してから再構成された軌道を模倣する、典型的な2段階ベースラインであるSfv [30] と詳細な比較を行います。元のSfv [30] はHOIや動物のモーションのようなドメイン向けに設計されていなかったため、公平な比較のためにモーション再構成ステージを適応させました。サッカーのドリブルデータセットについては、SLAHMR [56] を使用して3D人間ポーズを取得しました。ボールの3D位置は、バウンディングボックスの再投影誤差を最小化し、ボールの推定深度を組み込むことによって、その後再構成されました。動物のモーションデータについては、利用可能な事前データがないため、単純に逆運動学問題を解き、再投影誤差を最適化しました。表1の定量的比較結果は、ベースラインに対するアプローチの優れたパフォーマンスを示しています。提案手法は、より高い学習成功率と低いモーションジッター、および低い2D追跡誤差とオブジェクト2D追跡誤差を達成します。定性的な比較（図4）は、ベースラインの制限を示しています。Sfv*は、複雑なボールのターンアラウンドインタラクションを学習できず、不完全なトレーニングモーションのために、四足歩行の不自然な動きを示します。対照的に、提案手法は、2D参照に正確に一致するリアルなモーションを正常に合成します。

**汎用モーション追跡**
ビューに依存しない追跡ポリシーは汎用的であり、ビュー集約を介したゼロショットマルチビューモーション追跡を達成できます。追跡能力を徹底的に評価するために、AIST++データセットでアプローチをトレーニングし、グラウンドトゥルース3Dモーションを使用してトレーニングされた3Dモーション模倣ベースラインと比較します。表2に示すように、トレーニングされた単一視点追跡ポリシーは、未見のテストおよび生成されたモーションに正常に汎化します。さらに、提案されたビュー集約メカニズムは、ポリシーにマルチビュー追跡能力を与え、3Dモーション追跡エラーを低減して3Dモーション理解を向上させ、グラウンドトゥルースデータでトレーニングされた3Dベースラインと同等のパフォーマンスを達成します。

**4.3. 階層的制御の評価**
2Dモーションインターフェースを使用した階層的制御フレームワークの有効性を評価するために、AIST++およびサッカーのドリブルデータセットで2Dモーションジェネレーターをトレーニングします。定性的に、トレーニングされたモーションジェネレーターは、図5のさまざまなドリブルスキルの間の遷移によって示されるように、さまざまなモーションスキルをシームレスに組み合わせることによって複雑な軌道を合成する能力を正常に示します。さらに、提案された自己回帰型（AR）2Dモーションジェネレーターを、拡散ベースモデルベースライン [41] と比較してベンチマークしました。表3で詳述されているように、ARジェネレーターは、ベースラインと比較して、より低いFIDスコアとより高い成功率を達成し、よりリアルな2Dガイダンスを合成する上で優れたパフォーマンスを示します。

**4.4. 削減スタディ**
**データセットにおけるビューの多様性**
トレーニングデータにおけるビューの多様性から提案手法がどの程度効果的に利益を得られるかを調査するために、削減スタディを実施しました。多様な視点を持つ同じ3Dインスタンスから投影された2Dモーションを使用するポリシーの1つのバリアントと、均質な視点を持つ2Dモーションを使用する2番目のバリアントをトレーニングしました。図6に示すように、多様なビューモーションでトレーニングされたポリシーは、均質なビューモーションのみでトレーニングされたポリシーよりも大幅に自然な外観の振る舞いを合成します。特に、均質なビューポリシーは、挑戦的な「箱を持ち上げる」インタラクションを実行できず、複雑なスキルを学習するためのビュー多様性の重要性を強調しています。

**適応的状態初期化**
単一視点追跡トレーニングのための提案された適応的状態初期化戦略の有効性を評価します。図7に示すように、適応的状態初期化は、物理的に不可能な初期状態をより実現可能な状態に大幅に更新できます。

**5. 結論**

本研究では、ワイルドなビデオから抽出された2Dモーションシーケンスを直接模倣するために、ビューに依存しない追跡ポリシーをトレーニングすることにより、物理ベースのキャラクターコントローラーの学習にビデオデータを利用する新しいアプローチを提示します。フレームワークは、さまざまなモーションとキャラクターの関節構造にわたって堅牢に汎化し、3Dモーションデータにアクセスすることなく、複雑なHOIモーションと機敏な人間以外の動きを正常に学習します。さらに、ビュー集約を組み込むことにより、トレーニングされたポリシーは、グラウンドトゥルース3Dデータで直接トレーニングされたベースラインに匹敵する3D追跡パフォーマンスを達成します。最後に、ポリシーを提案された自己回帰型2Dモーションジェネレーターと統合することにより、物理的に妥当なモーションを生成できる階層コントローラーを確立し、さまざまな下流タスクに大きな可能性を示します。

**参考文献**

[1] Kevin Bergamin, Simon Clavet, Daniel Holden, and James Richard Forbes. Drecon: data-driven responsive control of physics-based characters. ACM Transactions On Graphics (TOG), 38(6):1–11, 2019. 3
[2] Jason Chemin and Jehee Lee. A physics-based juggling simulation using reinforcement learning. In Proceedings of the 11th ACM SIGGRAPH Conference on Motion, Interaction and Games, pages 1–7, 2018. 3
[3] Ching-Hang Chen, Ambrish Tyagi, Amit Agrawal, Dylan Drover, Rohith Mv, Stefan Stojanov, and James M Rehg. Unsupervised 3d pose estimation with geometric self-supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5714–5724, 2019. 2
[4] Stelian Coros, Philippe Beaudoin, and Michiel Van de Panne. Generalized biped walking control. ACM Transactions On Graphics (TOG), 29(4):1–9, 2010. 3
[5] Martin De Lasa, Igor Mordatch, and Aaron Hertzmann. Feature-based locomotion controllers. ACM transactions on graphics (TOG), 29(4):1–10, 2010. 3
[6] Zhiyang Dou, Xuelin Chen, Qingnan Fan, Taku Komura, and Wenping Wang. C·ase: Learning conditional adversarial skill embeddings for physics-based characters. In SIGGRAPH Asia 2023 Conference Papers, pages 1–11, 2023. 3
[7] Mohamed Hassan, Yunrong Guo, Tingwu Wang, Michael Black, Sanja Fidler, and Xue Bin Peng. Synthesizing physical character-scene interactions. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–9, 2023. 1, 3
[8] Jessica K Hodgins, Wayne L Wooten, David C Brogan, and James F O’Brien. Animating human athletics. In Proceedings of the 22nd annual conference on Computer graphics and interactive techniques, pages 71–78, 1995. 3
[9] Jordan Juravsky, Yunrong Guo, Sanja Fidler, and Xue Bin Peng. Padl: Language-directed physics-based character control. In SIGGRAPH Asia 2022 Conference Papers, pages 1–9, 2022. 3
[10] Jordan Juravsky, Yunrong Guo, Sanja Fidler, and Xue Bin Peng. Superpadl: Scaling language-directed physics-based control with progressive supervised distillation. In ACM SIGGRAPH 2024 Conference Papers, pages 1–11, 2024. 3
[11] Roy Kapon, Guy Tevet, Daniel Cohen-Or, and Amit H Bermano. Mas: Multi-view ancestral sampling for 3d motion generation using 2d diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1965–1974, 2024. 2, 7
[12] Seyoung Lee, Sunmin Lee, Yongwoo Lee, and Jehee Lee. Learning a family of motor skills from a single motion clip. ACM Transactions on Graphics (TOG), 40(4):1–13, 2021. 1
[13] Seunghwan Lee, Phil Sik Chang, and Jehee Lee. Deep compliant control. In ACM SIGGRAPH 2022 conference proceedings, pages 1–9, 2022. 1
[14] Yoonsang Lee, Sungeun Kim, and Jehee Lee. Data-driven biped control. In ACM SIGGRAPH 2010 Papers, New York, NY, USA, 2010. Association for Computing Machinery. 3
[15] Jiaman Li, Jiajun Wu, and C Karen Liu. Object motion guided human motion synthesis. ACM Transactions on Graphics (TOG), 42(6):1–11, 2023. 3
[16] Jiaman Li, C Karen Liu, and Jiajun Wu. Lifting motion to the 3d world via 2d diffusion. arXiv preprint arXiv:2411.18808, 2024. 2, 7
[17] Ruilong Li, Shan Yang, David A Ross, and Angjoo Kanazawa. Ai choreographer: Music conditioned 3d dance generation with aist++. In Proceedings of the IEEE/CVF international conference on computer vision, pages 13401–13412, 2021. 7
[18] Libin Liu and Jessica Hodgins. Learning basketball dribbling skills using trajectory optimization and deep reinforcement learning. ACM Transactions on Graphics (TOG), 37(4):1–14, 2018. 3
[19] Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for real-time simulated avatars. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10895–10904, 2023. 3
[20] Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris M Kitani, and Weipeng Xu. Universal humanoid motion representations for physics-based control. In The Twelfth International Conference on Learning Representations, 2023. 3
[21] Naureen Mahmood, Nima Ghorbani, Nikolaus F Troje, Gerard Pons-Moll, and Michael J Black. Amass: Archive of motion capture as surface shapes. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5442–5451, 2019. 3
[22] Denys Makoviichuk and Viktor Makoviychuk. rl-games: A high-performance framework for reinforcement learning. https://github.com/Denys88/rl_games, 2021. 2
[23] Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State. Isaac gym: High performance gpu-based physics simulation for robot learning, 2021. 7
[24] Jiageng Mao, Siheng Zhao, Siqi Song, Tianheng Shi, Junjie Ye, Mingtong Zhang, Haoran Geng, Jitendra Malik, Vitor Guizilini, and Yue Wang. Learning from massive human videos for universal humanoid pose control. arXiv preprint arXiv:2412.14172, 2024. 2, 3
[25] Meike Nauta, Doina Bucur, and Christin Seifert. Causal discovery with attention-based convolutional neural networks. Machine Learning and Knowledge Extraction, 1(1):19, 2019. 6
[26] Soohwan Park, Hoseok Ryu, Seyoung Lee, Sunmin Lee, and Jehee Lee. Learning predict-and-simulate policies from unorganized human motion data. ACM Transactions on Graphics (TOG), 38(6):1–11, 2019. 3
[27] Xue Bin Peng, Glen Berseth, and Michiel Van de Panne. Terrain-adaptive locomotion skills using deep reinforcement learning. ACM Transactions on Graphics (TOG), 35(4):1–12, 2016. 3
[28] Xue Bin Peng, Glen Berseth, KangKang Yin, and Michiel Van De Panne. Deeploco: Dynamic locomotion skills using hierarchical deep reinforcement learning. Acm transactions on graphics (tog), 36(4):1–13, 2017. 3
[29] Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel Van de Panne. Deepmimic: Example-guided deep reinforcement learning of physics-based character skills. ACM Transactions On Graphics (TOG), 37(4):1–14, 2018. 1, 3, 4, 5
[30] Xue Bin Peng, Angjoo Kanazawa, Jitendra Malik, Pieter Abbeel, and Sergey Levine. Sfv: Reinforcement learning of physical skills from videos. ACM Transactions On Graphics (TOG), 37(6):1–14, 2018. 2, 3, 5, 6, 8
[31] Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa. Amp: Adversarial motion priors for stylized physics-based character control. ACM Transactions on Graphics (ToG), 40(4):1–20, 2021. 3
[32] Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. Ase: Large-scale reusable adversarial skill embeddings for physically simulated characters. ACM Transactions On Graphics (TOG), 41(4):1–17, 2022. 3
[33] Huaijin Pi, Ruoxi Guo, Zehong Shen, Qing Shuai, Zechen Hu, Zhumei Wang, Yajiao Dong, Ruizhen Hu, Taku Komura, Sida Peng, et al. Motion-2-to-3: Leveraging 2d motion data to boost 3d motion generation. arXiv preprint arXiv:2412.13111, 2024. 2
[34] Marc H Raibert and Jessica K Hodgins. Animation of dynamic legged locomotion. In Proceedings of the 18th annual conference on Computer graphics and interactive techniques, pages 349–358, 1991. 3
[35] Daniele Reda, Hung Yu Ling, and Michiel Van De Panne. Learning to brachiate via simplified model imitation. In ACM SIGGRAPH 2022 conference proceedings, pages 1–9, 2022. 1
[36] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015. 3
[37] Agon Serifi, Ruben Grandia, Espen Knoop, Markus Gross, and Moritz Bacher. Robot motion diffusion model: Motion generation for robotic characters. In SIGGRAPH Asia 2024 Conference Papers, pages 1–9, 2024. 3
[38] Tianxin Tao, Matthew Wilson, Ruiyu Gou, and Michiel Van De Panne. Learning to get up. In ACM SIGGRAPH 2022 Conference Proceedings, pages 1–10, 2022. 3
[39] Chen Tessler, Yoni Kasten, Yunrong Guo, Shie Mannor, Gal Chechik, and Xue Bin Peng. Calm: Conditional adversarial latent models for directable virtual characters. In ACM SIGGRAPH 2023 Conference Proceedings, pages 1–9, 2023. 3
[40] Guy Tevet, Sigal Raab, Brian Gordon, Yoni Shafir, Daniel Cohen-or, and Amit Haim Bermano. Human motion diffusion model. In The Eleventh International Conference on Learning Representations, 2022. 3
[41] Guy Tevet, Sigal Raab, Setareh Cohan, Daniele Reda, Zhengyi Luo, Xue Bin Peng, Amit H Bermano, and Michiel van de Panne. Closd: Closing the loop between simulation and diffusion for multi-task character control. arXiv preprint arXiv:2410.03441, 2024. 3, 8
[42] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 6
[43] Marek Vondrak, Leonid Sigal, Jessica Hodgins, and Odest Jenkins. Video-based 3d motion capture through biped control. ACM Transactions On Graphics (TOG), 31(4):1–12, 2012. 3
[44] Bastian Wandt, James J Little, and Helge Rhodin. Elepose: Unsupervised 3d human pose estimation by predicting camera elevation and learning normalizing flows on 2d poses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6635–6645, 2022. 2
[45] Jiashun Wang, Yifeng Jiang, Haotian Zhang, Chen Tessler, Davis Rempe, Jessica Hodgins, and Xue Bin Peng. Hil: Hybrid imitation learning of diverse parkour skills from videos. arXiv preprint arXiv:2505.12619, 2025. 3
[46] Yinhuai Wang, Qihan Zhao, Runyi Yu, Hok Wai Tsui, Ailing Zeng, Jing Lin, Zhengyi Luo, Jiwen Yu, Xiu Li, Qifeng Chen, Jian Zhang, Lei Zhang, and Ping Tan. Skillmimic: Learning basketball interaction skills from demonstrations. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 17540–17549, 2025. 3
[47] Jungdam Won, Deepak Gopinath, and Jessica Hodgins. Physics-based character controllers using conditional vaes. ACM Transactions on Graphics (TOG), 41(4):1–12, 2022. 3
[48] Zhen Wu, Jiaman Li, Pei Xu, and C Karen Liu. Human-object interaction from human-level instructions. arXiv preprint arXiv:2406.17840, 2024. 3
[49] Zeqi Xiao, Tai Wang, Jingbo Wang, Jinkun Cao, Wenwei Zhang, Bo Dai, Dahua Lin, and Jiangmiao Pang. Unified human-scene interaction via prompted chain-of-contacts. In The Twelfth International Conference on Learning Representations, 2024. 3
[50] Kevin Xie, Tingwu Wang, Umar Iqbal, Yunrong Guo, Sanja Fidler, and Florian Shkurti. Physics-based human motion estimation and synthesis from videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11532–11541, 2021. 3
[51] Sirui Xu, Zhengyuan Li, Yu-Xiong Wang, and Liang-Yan Gui. Interdiff: Generating 3d human-object interactions with physics-informed diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14928–14940, 2023. 3
[52] Sirui Xu, Hung Yu Ling, Yu-Xiong Wang, and Liang-Yan Gui. Intermimic: Towards universal whole-body control for physics-based human-object interactions. arXiv preprint arXiv:2502.20390, 2025. 1, 3
[53] Yufei Xu, Jing Zhang, Qiming Zhang, and Dacheng Tao. Vitpose: Simple vision transformer baselines for human pose estimation. Advances in neural information processing systems, 35:38571–38584, 2022. 7
[54] Heyuan Yao, Zhenhua Song, Baoquan Chen, and Libin Liu. Controlvae: Model-based learning of generative controllers for physics-based characters. ACM Transactions on Graphics (TOG), 41(4):1–16, 2022. 3
[55] Heyuan Yao, Zhenhua Song, Yuyang Zhou, Tenglong Ao, Baoquan Chen, and Libin Liu. Moconvq: Unified physicsbased motion control via scalable discrete representations. ACM Transactions on Graphics (TOG), 43(4):1–21, 2024. 3
[56] Vickie Ye, Georgios Pavlakos, Jitendra Malik, and Angjoo Kanazawa. Decoupling human and camera motion from videos in the wild. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 8
[57] KangKang Yin, Kevin Loken, and Michiel Van de Panne. Simbicon: Simple biped locomotion control. ACM Transactions on Graphics (TOG), 26(3):105–es, 2007. 3
[58] KangKang Yin, Stelian Coros, Philippe Beaudoin, and Michiel van de Panne. Continuation methods for adapting simulated skills. ACM Trans. Graph., 27(3):1–7, 2008. 3
[59] Ri Yu, Hwangpil Park, and Jehee Lee. Human dynamics from monocular video with dynamic camera movements. ACM Transactions on Graphics (TOG), 40(6):1–14, 2021. 2, 3
[60] Ye Yuan and Kris Kitani. Ego-pose estimation and forecasting as real-time pd control. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10082–10092, 2019. 5
[61] Ye Yuan, Shih-En Wei, Tomas Simon, Kris Kitani, and Jason Saragih. Simpoe: Simulated character control for 3d human pose estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7159–7169, 2021. 3, 5
[62] Haotian Zhang, Ye Yuan, Viktor Makoviychuk, Yunrong Guo, Sanja Fidler, Xue Bin Peng, and Kayvon Fatahalian. Learning physically simulated tennis skills from broadcast videos. ACM Transactions on Graphics (TOG), 42(4):1–14, 2023. 2, 3
[63] Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Shaoli Huang, Yong Zhang, Hongwei Zhao, Hongtao Lu, and Xi Shen. T2m-gpt: Generating human motion from textual descriptions with discrete representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2

---

**補足資料**

**A. 強化学習の予備知識**
単一視点2Dモーション追跡に、状態 $S$、アクション $A$、遷移 $T$、報酬 $R$ で定義されるマルコフ決定プロセス（MDP）のタプル $M = (S, A, T, R)$ を用いてアプローチします。追跡コントローラーはポリシー $\pi(a_t|s_t)$ としてモデル化され、現在の状態 $s_t \in S$ に基づいてアクション $a_t \in A$ をサンプリングします。ポリシーのアクションに従ってシミュレーターで状態-アクション-報酬の軌跡 $\tau = \{(s_t, a_t, r_t)\}_{t=1}^T$ を収集し、シミュレーションダイナミクス $T(s_{t+1}|s_t, a_t)$ と報酬関数 $r_t = R(s_t, a_t)$ に従います。トレーニングの目的は、収集された軌跡全体で期待される累積リターン $R = \underset{\pi}{\mathbb{E}} [\sum_t \gamma^t r_t]$ を最大化する最適ポリシー $\pi^*$を学習することです。

**B. 実装の詳細**

**B.1. データ前処理**
再投影計算にはピンホールカメラモデルを採用し、各カメラビュー $C$ を $C = (R_C, \tau_C, f_C)$ でパラメータ化します。ここで $R_C$ と $\tau_C$ は外部回転と移動、$f_C$ は内部パラメータを表します。

ワイルドなビデオの場合、抽出された2Dモーションシーケンスから直接カメラパラメータ $C$ と初期状態 $s_0$ を推定します。内部パラメータを固定したまま、再投影損失を最小化することによってカメラ外部パラメータと3Dキャラクターポーズを共同で最適化します。結果として得られる3Dポーズは、単一視点追跡トレーニングの初期状態として使用されます。地面貫通を避けるために、ボディポイントが地面より下に落ちることを罰する追加の正規化項を含めます。

**B.2. 自己受容性状態表現**
自己受容性状態 $o_{prop}$ は、次の要素で構成されます。ルートの高さ $h_t \in \mathbb{R}$、ローカル座標系でのルート速度 $v_t \in \mathbb{R}^3$、ローカルフレームでのルート角速度 $w_t \in \mathbb{R}^3$、ローカルジョイント座標でのジョイント回転 $q_t \in \mathbb{R}^{6 \times J}$、ローカルジョイント速度 $q̇_t \in \mathbb{R}^{3 \times J}$、およびローカルルートフレームでの主要ボディリンクのカルテジアン位置 $p_t \in \mathbb{R}^K$。

**B.3. 適応的状態初期化アルゴリズム**
提案された適応的状態初期化戦略の詳細をアルゴリズム1に示します。

```
アルゴリズム 1 初期状態分布更新とサンプリング
1: 入力: 初期状態分布 d(s0), 参照モーションフレーム X = {x0, ..., xT}, Criticネットワーク Vϕ
2: 出力: 更新された初期状態バッファ B
3: 初期化: 各 Bt がフレーム xt 専用のバッファ B = {Bt}T t=0 を作成する
4:
5: function SAMPLEINITIALSTATE(Bt)
6:   Scores ← {Score(s) | s ∈ Bt}
7:   Probabilities ← Normalize(exp(β · Scores)) ▷ βは優先度を制御する
8:   return Sample(Bt, Probabilities)
9: end function
10:
11: function UPDATESTATEBUFFER(B, Rollouts)
12:   for each state {st, xt} at frame xt in Rollouts do
13:     CriticScore ← Vϕ(st) ▷ 値/追跡パフォーマンスを推定する
14:     if CriticScore > Threshold or |Bt| < MinSize then
15:       Score(st) ← CriticScore
16:       Bt ← Bt ∪ {st} ▷ フレーム t のバッファに状態を格納する
17:       if |Bt| > MaxSize then
18:         Bt ← RemoveLowestScore(s', Bt) ▷ バッファサイズ制限を維持する
19:       end if
20:     end if
21:   end for
22: end function
```

**B.4. 2Dモーション・トークナイザー**
モーション・トークナイザーとしてVQ-VAEを使用します。VQ-VAEは、エンコーダー $E(\cdot)$とデコーダー $D(\cdot)$で構成されます。エンコーダーは、相対2Dモーションシーケンス $x_{rel}$ を、低次元の潜在ベクトル $z = [z_0, z_1, ..., z_{T/l}]$ のシーケンスに変換します。ここで $l$ は時間的ダウンサンプリング係数です。各潜在ベクトル $z_i$ は、学習された離散埋め込みのコードブック $C = \{c_k\}_{k=1}^K$ の最も近いエントリに量子化されます。$\hat{z}_i = \arg \min_{c_k \in C} \|z_i - c_k\|^2$。デコーダーは、量子化された潜在表現から元の相対2Dモーションを再構成します。つまり、$x_{rel} = D(\hat{z})$ です。ここで $D(\cdot)$ はデコーダー、$\hat{z}$ は量子化された潜在ベクトルです。

VQ-VAEは、次の損失関数を最小化することによってトレーニングされます。

$L_{VQ} = L_{recon} + \|sg[Z] - \hat{Z}\|^2_2 + \beta\|Z - sg[\hat{Z}]\|^2_2 \quad (9)$

ここで、$L_{recon}$は再構成損失、$sg[Z]$はストップグラディエント演算子、$ \hat{Z}$は量子化された潜在ベクトル、$ \beta$はコミットメント損失のバランスをとるハイパーパラメータです。コードブック埋め込み [63] の指数移動平均（EMA）更新を通じてトレーニングを安定させます。

**B.5. 自己回帰型ジェネレーター**
トークンを予測するために、自己回帰型ジェネレーターにトランスフォーマーデコーダーアーキテクチャを使用します。各離散トークンインデックスは最初に埋め込みにマッピングされ、条件付け変数（指定された制御入力から導出されるか、条件なし生成のためにゼロに設定される）がシーケンスの前に付けられます。絶対位置埋め込みが追加され、時間的順序がエンコードされます。

単一視点2D入力の深度の曖昧さに敏感なダンスモーション生成のような生成タスクの場合、追加のビューポイントを提供して、欠落した深度キューを解決するのに役立つマルチビュー2Dジェネレーターをトレーニングします。マルチビュージェネレーターのトレーニングデータを作成するために、2Dシーケンスを追跡しているときのシミュレートされたキャラクターからの3Dモーション状態を収集し、複数の2Dビューに投影して、ペア化されたマルチビューモーションサンプルを取得します。結果のマルチビュー2Dモーションを、各フレームがビューIDで順序付けされた複数のモーション・トークンを含むシーケンシャルデータとして表現します。

**B.6. 主要ハイパーパラメータ**
ポリシー学習の主要なハイパーパラメータを表4に示します。PPO実装はRLGames [22]に基づいています。VQ-VAEおよび自己回帰型ジェネレーターのハイパーパラメータを表5および表6に示します。

**5. 結論**

本研究では、ワイルドなビデオから抽出された2Dモーションシーケンスを直接模倣するビューに依存しない追跡ポリシーをトレーニングすることにより、物理ベースのキャラクターコントローラーを学習するための新しいアプローチを提示します。フレームワークは、さまざまなモーションとキャラクターの関節構造にわたって堅牢に汎化し、3Dモーションデータにアクセスすることなく、複雑なHOIモーションと機敏な人間以外の動きを正常に学習します。さらに、ビュー集約を組み込むことにより、トレーニングされたポリシーは、グラウンドトゥルース3Dデータで直接トレーニングされたベースラインに匹敵する3D追跡パフォーマンスを達成します。最後に、ポリシーを提案された自己回帰型2Dモーションジェネレーターと統合することにより、物理的に妥当なモーションを生成できる階層コントローラーを確立し、さまざまな下流タスクに大きな可能性を示します。

**参考文献** (一部省略)

**補足資料**

**C. 追加結果**

**状態初期化戦略**
提案された適応的状態初期化戦略の有無による、非常に困難な動物のジャンプモーションにおける平均追跡期間の学習曲線と比較します。図8に示すように、提案手法はポリシー学習の効果的な収束を可能にする一方、物理的に妥当でない初期状態がトレーニングプロセスを妨げるため、比較対象は苦労します。

**マルチビューガイダンスの影響**
ビューに依存しない2D追跡ポリシーは、多様なトレーニングビューポイントを利用して堅牢な3D理解を獲得しますが、単一視点入力の固有の深度の曖昧さは、複雑なモーションの推論中に依然として残る可能性があり、不規則な動きを引き起こします。これを調査するために、提案された2Dモーションジェネレーターによって合成された2つの直交する2D参照ビューを使用して実験を行いました。これらのビューのうち1つのみを追跡する場合と、両方を同時に追跡する場合のポリシーのパフォーマンスを比較しました。図9に示すように、単一の生成ビューを追跡するポリシーは、深刻な深度の曖昧さに苦しみ、足の滑りなどのアーティファクトが発生します。対照的に、両方のビューを統合するポリシーは、この曖昧さを正常に解決し、安定したポーズと妥当なモーションを維持します。

**D. 制限と今後の作業**
現在のフレームワークでは、2D追跡ポリシーは再投影誤差ベースの報酬にのみ依存しています。この幾何学的な目的は、精巧な操作やツールの使用のような、細かいオブジェクトインタラクションに必要な正確な接触ダイナミクスを学習するには不十分な場合があります。将来の研究では、これらの接触制約を明示的に固定するための追加のセマンティック報酬を導入することにより、これを対処できる可能性があります。さらに、現在のフレームワークでは、カメラパラメータが合理的に正確に推定されていると仮定しています。しかし、実際には、不正確なカメラ推定は、2D追跡ポリシーのパフォーマンスを大幅に低下させる可能性があります。これを軽減するために、将来のイテレーションでは、カメラパラメータを学習プロセスに直接組み込み、モーション制御とカメラ推定の共同最適化を可能にすることが考えられます。