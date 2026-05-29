# Multi-agent Matching

複数エージェントが同じ環境内の物体を観測したときに、「どの検出が同じ実物体を指しているか」を推定するためのシミュレーションです。  
3台のエージェント、複数の物体、自己位置推定誤差、物体検出ノイズ、誤検出を含む観測を生成し、以下の3種類の対応付け手法を比較します。

- RoCo-style pairwise matching
- Pairwise graph matching with RRWM
- 3-wise / k-wise MGM style RRWM
- Partial optimal transport rigid registration

実験は [notebooks/experiment.ipynb](notebooks/experiment.ipynb) から実行します。結果は repository root、つまりこの `research` フォルダ内の `results/YYYYMMDD_HHMMSS_JST_seed_X/` に保存されます。CSV は解析用に保存し、同時に `summary.html` も生成して、画像と表をブラウザで確認できるようにしています。

## 問題設定

各エージェントは同じ global 座標系上に存在する物体群を、それぞれの local 座標系から観測します。実際のマルチエージェント認識では、各エージェントの自己位置推定に誤差があり、物体検出にも位置・向きのノイズや誤検出が含まれます。

このシミュレーションでは、各エージェントの検出結果をいったん noisy な自己位置で global 座標へ投影し、その global 推定位置・姿勢だけを使って物体対応を推定します。評価時だけ `true_obj_id` を使い、推定された対応が本当に同じ物体かどうかを判定します。

基本の設定は次の通りです。

- エージェント数: 3
- 物体数: 12
- 環境サイズ: 60 m x 40 m
- センサ範囲: 32 m
- 視野角: 150 degrees
- 検出確率: 0.85
- 物体位置ノイズ標準偏差: 0.7 m
- 物体 yaw ノイズ標準偏差: 4 degrees
- エージェント自己位置ノイズ標準偏差: 1.0 m
- エージェント yaw ノイズ標準偏差: 3 degrees
- 誤検出: 各エージェント2個

## True Scene の構築

True Scene は [src/simulation.py](src/simulation.py) の `generate_true_scene` で作ります。

3台のエージェントは固定配置です。

| agent | x | y | theta |
|---|---:|---:|---:|
| A0 | -18.0 | -8.0 | 15 deg |
| A1 | 18.0 | -6.0 | 165 deg |
| A2 | 0.0 | 16.0 | -90 deg |

物体は指定された矩形領域内にランダム配置します。各物体には `obj_id`, `x`, `y`, `theta`, `length`, `width` があり、車両のような長方形物体を想定しています。乱数 seed を固定することで、同じ条件の実験を再現できます。

True Scene の図では、真の物体を四角、真のエージェントを三角で描画し、それぞれに向きを表す矢印を付けます。

## Observed Scene の構築

Observed Scene は [src/simulation.py](src/simulation.py) の `generate_detections` で作ります。

流れは次の通りです。

1. 各エージェントの真の pose に位置・yaw ノイズを加え、noisy な自己位置推定 `agent_pose_est` を作る。
2. 各物体の真の global 位置を、エージェント基準の local 座標に変換する。
3. センサ範囲と視野角に入った物体だけを検出候補にする。
4. 検出確率に従って一部の候補を検出漏れにする。
5. local 座標上の物体位置と yaw に検出ノイズを加える。
6. noisy な自己位置推定を使って、local 検出を global 座標へ投影する。
7. 実在しない物体として outlier 検出を追加する。

各 Detection には、評価用の `true_obj_id` と、推定に使う `x_global_est`, `y_global_est`, `theta_global_est` が入ります。`true_obj_id = -1` は outlier です。

Observed Scene の図では、真のエージェント位置、noisy な自己位置、各エージェントの検出点を重ねて描きます。これにより、自己位置誤差と検出誤差によって、同じ物体由来の検出点が完全には重ならない様子を確認できます。

## 実装している手法

### RoCo-style pairwise matching

RoCo-style は2エージェント間の検出を直接対応付ける pairwise 手法です。

まず、検出点同士の距離から初期対応候補を作ります。次に、各検出の近傍構造を比較します。具体的には、ある検出を中心にした近傍物体への相対 pose 変換を計算し、別エージェント側の近傍構造とどれだけ整合するかを edge similarity として評価します。

最終的な similarity は、検出点同士の距離 similarity と近傍構造 similarity を組み合わせて作り、Hungarian algorithm で1対1対応に離散化します。

特徴:

- 2エージェントごとに独立して対応付ける。
- 局所的な周辺構造を使う。
- 実装が比較的軽く、結果の解釈がしやすい。

### Pairwise graph matching with RRWM

Pairwise RRWM も2エージェント間の対応を解きますが、対応候補を graph matching 問題として扱います。

検出 `di` と `dj` の組を1つの対応候補とし、候補ごとに unary affinity を計算します。unary affinity は、位置差と yaw 差が小さいほど高くなります。さらに、2つの対応候補が同時に成立するかを pairwise affinity で評価します。これは、エージェント i 側の2検出間の相対 pose と、エージェント j 側の2検出間の相対 pose がどれだけ一致するかを見るものです。

その affinity matrix に対して RRWM を実行し、soft score を得ます。最後に Hungarian algorithm で1対1対応に変換します。

特徴:

- 候補単体の近さと、候補同士の構造整合性を同時に使う。
- RoCo-style より graph matching として明示的。
- ただし pairwise なので、3台全体の整合性は直接は最適化しない。

### 3-wise / k-wise MGM style RRWM

k-wise RRWM は3エージェントを同時に扱います。候補は `(A0の検出, A1の検出, A2の検出)` の三つ組です。

三つ組候補の unary affinity は、3つの検出点が互いに近く、yaw も整合しているほど高くなります。さらに、2つの三つ組候補の間で、各エージェント内の相対 pose 構造が一貫しているかを pairwise structure affinity として評価します。

RRWM で三つ組候補の soft score を最適化した後、同じ検出が複数の三つ組に使われないよう greedy に離散化します。比較表では、この3者対応を `(0,1)`, `(0,2)`, `(1,2)` の pairwise 対応に分解して、他の2手法と同じ形式で評価します。

特徴:

- 3エージェントの整合性を同時に見る。
- pairwise 手法では曖昧な対応を、多者制約で抑えられる可能性がある。
- 候補数が増えやすく、計算量は pairwise より重くなりやすい。

### Partial optimal transport rigid registration

Partial OT は、Qin et al. の "Rigid Registration of Point Clouds Based on Partial Optimal Transport" をこの問題設定向けに 2D 化した手法です。

anchor agent の global 検出点群と target agent の local 検出点群を点群登録問題として扱い、range constraint 付き transport plan と weighted Procrustes による剛体変換推定を交互に解きます。transport plan の total mass 上限 `beta2` によって、外れ値や片方にしか見えていない検出を無理に対応させないようにします。

実装の入口:

- [src/graph_matching.py](/Users/yusei/Documents/research/src/graph_matching.py): `partial_ot_rigid_registration_2d`, `partial_ot_pairwise_matching`, `partial_ot_anchor_pose_registration`
- [src/roco_iterative.py](/Users/yusei/Documents/research/src/roco_iterative.py): `run_iterative_partial_ot_pose_adjustment`

主要パラメータ:

- `epsilon`: transport kernel の広がり。大きいほど大域的、小さいほど局所的な対応になる。
- `epsilon_decay`: outer iteration ごとの `epsilon` 減衰率。
- `beta2`: total transported mass の上限。重なり率や外れ値率に合わせて調整する。
- `outer_iter`, `inner_iter`: 剛体変換更新と transport plan 更新の反復回数。

## Iterative pose adjustment

対応付けだけでなく、対応結果を使った pose adjustment も実装しています。対象は RoCo-style、Pairwise RRWM、k-wise RRWM、Partial OT の4手法です。

反復処理の流れは次の通りです。

1. 現在のエージェント pose 推定で、local 検出を global 座標へ投影し直す。
2. 選択した matching 手法で全エージェントペアの対応を求める。
3. anchor agent を固定し、anchor との対応点から他エージェントの pose を推定する。
4. damping をかけて pose を更新する。
5. pose 更新量が閾値以下になるか、最大反復回数に達するまで繰り返す。

この反復結果から、iteration ごとの matching 精度、pose 更新量、agent/object の pose error を CSV と HTML レポートに保存します。

## 図の作り方

図は [src/visualization.py](src/visualization.py) で作ります。

- `True_scene_seed_X.png`: 真の物体と真のエージェント配置。
- `Observed_scene_seed_X.png`: noisy な自己位置と検出点。
- `RoCo_ij_seed_X.png`: RoCo-style の pairwise matching 結果。
- `Pairwise_ij_seed_X.png`: Pairwise RRWM の matching 結果。
- `kwise_ij_seed_X.png`: k-wise 結果を pairwise に分解した matching 結果。
- `Iterative_*_adjusted_scene_seed_X.png`: pose adjustment 後の観測シーン。
- `Iterative_*_ij_seed_X.png`: pose adjustment 後の matching 結果。

pairwise matching 図では、同じ真の物体に対応している線を TP、誤対応を FP、本来対応すべきだったが未検出の対応を FN として描き分けます。図中には precision / recall も表示します。必要に応じて真の物体位置も星印で重ね、推定された対応と真値の位置関係を見られるようにしています。

## 3手法の比較方法

各手法の結果は `true_obj_id` を使って評価します。推定時には `true_obj_id` は使いません。

pairwise 評価では、各エージェントペアについて以下を数えます。

- TP: 同じ `true_obj_id` を持つ検出同士を対応付けた数
- FP: 異なる物体、または outlier を対応付けた数
- FN: 両エージェントに見えていたのに対応付けられなかった真の対応数
- precision: `TP / (TP + FP)`
- recall: `TP / (TP + FN)`

k-wise 手法は3者対応として評価した後、3手法の比較表では pairwise に分解して `RoCo-style` と `Pairwise RRWM` と同じ `pair = 0-1, 0-2, 1-2` の単位で比較します。比較結果は `matching_comparison.csv` に保存され、`summary.html` でも表として確認できます。

pose adjustment については、agent pose と object pose の推定値を真値と比較し、位置誤差、yaw 誤差、両者を足した pose error を出します。summary では overall / agents / objects の平均誤差や RMSE を確認できます。

## 出力

実験を実行すると、`results/YYYYMMDD_HHMMSS_JST_seed_X/` に以下が保存されます。

- PNG: シーン図、matching 図、pose adjustment 後の図
- CSV: matching 比較、反復履歴、pose error、推定物体位置
- HTML: `summary.html`

`summary.html` は、同じフォルダ内の PNG と CSV をまとめて表示する簡易レポートです。CSV は消さず、解析しやすい元データとして残します。

`results/` は `.gitignore` に含めているため、シミュレーション結果は GitHub にはアップロードしません。

## 実行方法

1. `notebooks/experiment.ipynb` を開く。
2. 上から順にセルを実行する。
3. 表示された `output_dir` に移動する。
4. `summary.html` をブラウザで開き、図と表を確認する。

主要なパラメータは notebook 内の `roco_params`, `rrwm_params`, `kwise_params`, `roco_iterative_params` で調整できます。
