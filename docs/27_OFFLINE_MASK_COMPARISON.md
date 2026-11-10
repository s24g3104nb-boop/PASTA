# 乾麺画像のマスク比較実験（本番未採用）

この工程は既存アプリから独立した画像処理の比較です。UI、測定器、SQLite、食事日記、既存マスク、学習モデルを変更しません。重量を予測せず、モデル学習・マスク承認・データセット統合も行いません。

## 入力とデータ保護

受領済みの `originals/labels.csv` と画像原本を読みます。20/40/60/80/100g × 3撮影状態の15枚、空欄でない単一の物理試料系列、画像ID・パスの一意性、SHA-256を実行前に確認します。同一試料を追加して撮影した画像を、学習と評価に分割しません。

全ファイル・付属ラベル・生成画像・集計結果・実測校正値は私用の `pasta_weight_app_v2/data/` 以下へ保存してください。既存出力先への上書きは拒否します。アプリ等の書込みを停止し、作業前に既存データのハッシュ台帳を作り、作業後に照合します。

```python
from pathlib import Path
import json
from pasta_weight_app_v2.experiments.intake_receipt import _snapshot, verify_protection

root = Path('pasta_weight_app_v2/data')
before = _snapshot(root)
output = root / 'NEW_MASK_EXPERIMENT'
output.mkdir(exist_ok=False)
(output / 'protection_before.json').write_text(json.dumps(before), encoding='utf-8')
# 比較処理後に実行
assert verify_protection(output)['unchanged']
```

この照合は既存ファイルの変更・欠落を検出するもので、バックアップの代用ではありません。本工程はDB移行を行いません。

## 比較条件

EXIF向き補正と最長辺1400pxの入力処理は既存検出器と共通です。ベースラインのパラメータは画面側の `DEFAULT_SPREAD` / `DEFAULT_BUNDLE` を読みます。端面検出関数のデフォルト値と画面の設定値を混同しません。開発設定で利用者が上書きした状態を再現する実験ではありません。

|方式|処理と比較上の意味|
|---|---|
|hsv_baseline|既存HSV＋3×3 closing。面積・連結成分・既存本数候補を維持|
|hsv_no_close|closingを外し、近接領域の結合による変化を観察|
|lab_only|8bit Labの黄方向・赤方向・明度の固定閾値|
|hsv_lab|HSVとLabの共通領域。背景抑制と麺の欠落を同時に確認|
|shadow_clahe|Lab明度の局所補正後に色条件を適用。真の反射除去ではない|
|opening|HSV＋Labマスクを3×3 opening。細い麺も消える可能性あり|
|watershed|HSV＋Labの画素集合を保ち、512pxで距離変換の局所極大から分割|
|manual_roi|人手の粗い領域指定とHSV＋Lab。自動手法とは区別する|

閾値は重量に対する最適化で選びません。全画面の特徴量は麺以外を含み得ます。Watershedの分割領域数は麺の本数ではありません。低角度や手持ちの端面では、小径・短い中心間隔の局所円検出も比較しますが、正解本数がないため優劣は確定できません。

手動注釈は画像IDをキーとするJSONです。各項目に `roi_polygon_normalized`（0〜1の頂点）、`positive_points`、`negative_points`、`observed_view` を記録します。ROIは完全な正解マスクではなく、少数の参照点も画像全体の精度評価には使えません。見直し時は新しいファイル名で履歴を保存してください。端面が確認できない側面画像のROIは空配列とし、特徴量は未対応・nullとします。

```sh
.venv/bin/python -m pasta_weight_app_v2.experiments.compare_masks \
  --originals pasta_weight_app_v2/data/RECEIPT/originals \
  --annotations pasta_weight_app_v2/data/NEW_MASK_EXPERIMENT/roi_draft.json \
  --output pasta_weight_app_v2/data/NEW_MASK_EXPERIMENT/results
```

100%、75%、50%のデジタル縮小を処理し、計360行を出力します。縮小による面積減少の大部分は画素数の減少です。面積比と円候補数の変動を別々に確認してください。これは実際の撮影距離やレンズ・遠近を変えた検証ではありません。

## 集計と可視化

`index.html` は各画像の元画像・マスク・輪郭と円・ベースラインとの差分を比較します。差分の緑は追加画素、赤は除去画素であり、正誤の印ではありません。`features.csv` と `comparison.json` に面積、面積率、回転外接矩形の長短軸、輪郭長、連結領域数、円候補、骨格長・幅・密度を保存します。骨格値は512px座標上であり、実際の麺長・麺径ではありません。

集計スクリプト `experiments/summarize_masks.py` はNumPyとMatplotlibを使います。描画用の環境はアプリの環境と分離しても構いません。`--help` で引数を確認してください。出力の数値一覧・相関・縮小比較も私用データです。

同一系列5重量の相関は記述統計です。相関が高くても、正解マスクに近いことや、独立した重量推定精度を示しません。背景面積・手の面積・画角が重量と一緒に変化する可能性があります。MAE・RMSE等は正当な重量予測と独立評価が整うまで算出しません。

## 実製品の校正値の受け皿

```sh
.venv/bin/python -m pasta_weight_app_v2.experiments.product_calibration --template
.venv/bin/python -m pasta_weight_app_v2.experiments.product_calibration PRIVATE_PROFILE.json
```

空のテンプレートへ製品名、太さmm、長さmm、実測本数、合計重量g、秤の分解能gを入力します。平均1本重量は合計重量÷本数です。数値が未入力・非有限・負数、ゼロ本、重複JSONキー等なら拒否します。1g/本等の仮の初期値はありません。

出力は未承認の校正ドラフトで、本番DBには登録しません。分解能由来の幅は「正味重量の表示が最近接目盛りへ丸められた」と仮定した丸めだけの幅です。秤精度、風袋、数え間違い、製品のばらつき等を含む総合的な誤差範囲ではありません。折れていない同じ製品・同じ長さの乾麺を数え、反復秤量と秤の仕様を別途確認してください。校正が正しくても、画像の検出本数が正しくなるわけではありません。

## 検証と次工程の条件

```sh
.venv/bin/python -m pytest -q pasta_weight_app_v2/tests
.venv/bin/python tools/check_public_tree.py --history
git diff --exit-code
```

合成画像・一時DBを用いるテストは、現行マスク／円候補との一致、原本保護、上書き拒否、系列混在拒否、未対応値の区別、校正検証を確認します。実機の重量精度を保証しません。Windowsは `.venv/bin/python` を `.venv\Scripts\python.exe` に置き換え、コマンドを一行で実行できます。

本番採用の前に、撮影状態別の正解マスクと見えている端面の正解、実製品の校正値、独立した物理試料・撮影セッションを確保してください。モデルや処理の選択に使った系列を最終評価へ再利用せず、重量別・状態別の誤差、最大誤差、出力を拒否した割合も確認します。手持ち側面や重なりで見えない麺については、単眼画像から回復できない情報が残ります。
