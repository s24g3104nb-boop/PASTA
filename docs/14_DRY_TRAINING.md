# 乾麺写真の取込・学習・評価

既存のOpenCV本数検出器を残し、比較用にNumPyだけのCPU画像特徴量＋リッジ回帰を追加した。深層学習済みモデルではない。私有写真のプレビュー42枚は受領済みだが、全件の実測重量が未確認のため実写真による学習・評価は未実施。候補のグループ名から重量を登録しない。5重量すべてが揃うとも確認されていない。

以下は **PASTAリポジトリのルート**で実行する。Windowsでは `.venv/bin/python` を `.venv\Scripts\python` に読み替える。写真、ラベル、モデル、結果はGitに登録しない `pasta_weight_app_v2/data/` 内で管理する。

## 1. 写真とCSV

`dataset/ml_template.csv` を `pasta_weight_app_v2/data/incoming/labels.csv` にコピーし、同じディレクトリ内に写真を置く。元の `dataset_template.csv` を使う場合は `source` 列を追加する。

各行は `image_path`（例 `photos/20g_01.jpg`）、`batch_id`（同じ秤量試料の識別子）、`session_id`（同じ撮影セッション）、`product_id`、`mode`（`spread`/`bundle_end`（`bundle`も同義として受理））、`actual_weight_g`（有限の正の実測g）、`source`（`real`/`synthetic`）が必須。ほかの既存列は任意。連写・同じ麺の置き直し・増量前後で共通する撮影条件は追跡できるよう記録する。束のOpenCV評価は端面の正面写真のみ。横からの束にも回帰入力は可能だが、その条件を独立に収集・評価する必要がある。

写真のファイル名から重量は推測しない。20/40/60/80/100gは実測ラベルで入力する。画像中の秤表示・重量を示す文字は写さないか、事前に一貫した方法で切り落とす。重量ごとに背景を変えるなど、答えと撮影条件の結び付きを避ける。写真を受領したら重量帯別枚数・製品・モード・独立秤量数・距離・照明・欠損をまず確認する。

```bash
.venv/bin/python -m pasta_weight_app_v2.pasta_ml validate pasta_weight_app_v2/data/incoming/labels.csv
.venv/bin/python -m pasta_weight_app_v2.pasta_ml import pasta_weight_app_v2/data/incoming/labels.csv --destination pasta_weight_app_v2/data/experiment_001
```

取込先は未存在のパスを指定する。元写真は変更しない。取込時にJPEG/PNGの実形式、サイズ（30MB/3000万画素以下）、EXIF向き、破損、相対パスを検査し、管理用PNGをEXIFなしで保存する。同一画像の矛盾ラベルは拒否する。個人情報の写り込みは人手で確認する。

## 2. 分割を固定

```bash
.venv/bin/python -m pasta_weight_app_v2.pasta_ml split pasta_weight_app_v2/data/experiment_001/dataset.csv --output pasta_weight_app_v2/data/experiment_001/split.csv --seed 42
.venv/bin/python -m pasta_weight_app_v2.pasta_ml validate pasta_weight_app_v2/data/experiment_001/split.csv
```

同一batch、session、正規化画素一致、近似画像（64bit dHash距離4以下）を推移的に同じグループへ束ねる。独立3グループ未満では停止。グループを固定seedで約70/15/15%に分け、各集合を空にしない。近似判定は安全側の簡易検査で、類似画像の完全検出を保証しない。非常に似た写真が別秤量でも同一グループになることがある。分割後に画像を人手で見比べ、必要なら共通batch/sessionを付けて元CSVから再作成する。既存splitの再分割は禁止。撮影日・製品ホールドアウト実験は別の実験として設計する。

割合・各重量/モード/製品の被覆はデータ依存であり、自動的には保証しない。訓練では **モード×製品ごとに独立した訓練グループ3つ以上・重量2種類以上**を要求する。最低枚数は精度の保証ではない。5枚だけ、または一度の撮影だけでは不足し得る。両モード×各重量について独立秤量と異なる撮影条件を増やし、十分なval/testを確保する。枚数目標は最初の評価結果と失敗条件を見て決める。

## 3. OpenCVの比較基準

実製品を校正し、例えば `pasta_weight_app_v2/data/experiment_001/calibration.csv` を作る。

```csv
product_id,g_per_strand
my_product,0.93
```

`0.93` は形式を示す架空値。自分の製品で数えた複数本の実測重量÷本数を入力する。製品未校正・検出0本は失敗として総数に含める。

```bash
.venv/bin/python -m pasta_weight_app_v2.pasta_ml evaluate pasta_weight_app_v2/data/experiment_001/split.csv --calibrations pasta_weight_app_v2/data/experiment_001/calibration.csv --split val --output pasta_weight_app_v2/data/experiment_001/opencv_val.json
```

## 4. 実験用回帰を明示的に訓練

```bash
.venv/bin/python -m pasta_weight_app_v2.pasta_ml train pasta_weight_app_v2/data/experiment_001/split.csv --output pasta_weight_app_v2/data/experiment_001/model_v1.json
.venv/bin/python -m pasta_weight_app_v2.pasta_ml evaluate pasta_weight_app_v2/data/experiment_001/split.csv --model pasta_weight_app_v2/data/experiment_001/model_v1.json --split val --output pasta_weight_app_v2/data/experiment_001/model_val.json
```

画像のRGB統計・HSVヒストグラム・黄色領域の分布・エッジ量から重量gを回帰する。標準化と係数はtrainだけで推定、val結果をモデルに記録する。重量ラベルは特徴量に使わない。testで正則化や前処理を調整しない。必要なら `--alpha` を **valだけ**で比較し、決めた後に最終評価する。モデルJSONは版・日時・train/全データハッシュ・source・係数・対応モード/製品・重量範囲・val指標を保存し、pickleは使わない。

出力は20/40/60/80/100gの5択に丸めず、**35gや75gなども表す連続値**である。ただし5重量帯だけを学習して35g・75gを正しく補間できるかは別の検証課題。35g・75gをそれぞれ実際にはかりで用意し、独立した秤量・撮影セッションで撮った未使用写真を最終testに確保する。その重量別MAE・偏り・成功率で確認し、20g刻みの写真の評価だけから中間重量の精度を主張しない。20g未満・100g超など訓練重量範囲外を推定可能とは扱わない。

realとsyntheticの混在を拒否する。合成画像で実装を試すときだけ別CSV・別モデルに `--allow-synthetic` を付ける。その数値は実物性能ではない。写真がないCSVや空CSVでは学習せずエラーで終了する。

## 5. 最後にtest、1枚推定

```bash
.venv/bin/python -m pasta_weight_app_v2.pasta_ml evaluate pasta_weight_app_v2/data/experiment_001/split.csv --model pasta_weight_app_v2/data/experiment_001/model_v1.json --split test --final-test --output pasta_weight_app_v2/data/experiment_001/model_test.json
.venv/bin/python -m pasta_weight_app_v2.pasta_ml predict pasta_weight_app_v2/data/new_photo.jpg --model pasta_weight_app_v2/data/experiment_001/model_v1.json --mode spread --product-id my_product
```

test開封は明示指定が必要。同じモデルのtest二重評価は隣接する `.test-used.json` で防ぐ。調整して作り直したモデルでも、同じtestを何度も見て設計を変えればリークになる。最終テストを開封後の開発では別の未使用実写真を集める。

評価CSVはモデル記録のデータ・ラベル・分割ハッシュと一致する必要がある。各評価は全件・成功件数・失敗件数・成功率、成功画像のMAE/RMSE/MAPE/中央値絶対誤差/符号付き平均誤差を出し、mode/製品/重量/照明/背景別も併記。全失敗時の誤差は `null` とし、0とはしない。失敗画像を成功スコアから隠さない。未学習モード/製品、訓練特徴範囲外、予測重量範囲外では推定を棄却し再撮影・追加データを求める。特徴範囲内だから正しいという保証や校正済みの信頼区間はない。

CLIの単画像推定に加えて、[AI学習管理画面](17_TRAINING_MANAGEMENT.md) から実写モデルを学習・選択し、測定画面でOpenCVと回帰を切り替えられる。UI管理の製品IDはパスタデータの数値IDを文字列にしたもの。CLIの任意製品コードはCLI内で一貫させる。完成料理の種類分類は別モデル・別データであり、この回帰は完成料理の重量/総カロリーを出さない。

## 6. 写真を追加して再学習する

既存の `experiment_001/`（写真・元CSV・split・モデル・結果）はそのまま残す。写真を追加するときは新しい `data/incoming_round2/` に写真とラベルCSVを用意し、これまでの画像も含めた `labels.csv` を作る。旧画像のbatch/session識別子とラベルは変更しない。`split` 列を付け、旧train/valの割当を保持したまま、新しい独立グループをtrain/val/testへ割り当てる。割当は画像行単位ではなくbatch/session/近似画像グループ単位にする。必要なら元画像から同じ相対フォルダ構成でコピーし、各行のimage_pathを新しいCSVから辿れるようにする。

旧testをまだ開封していない場合、その写真は引き続きtestに固定し、train/valへ移さない。旧testの結果を既に見て次の開発方針を変えた場合、**旧testは今回の訓練・調整・最終評価用CSVから除外**して旧実験の記録として残し、別の独立した未使用写真を新testとして確保する。35g/75gの補間検証用写真もこの条件で集める。既存splitを消して全画像をランダムに再分割しない。

```bash
.venv/bin/python -m pasta_weight_app_v2.pasta_ml validate pasta_weight_app_v2/data/incoming_round2/labels.csv
.venv/bin/python -m pasta_weight_app_v2.pasta_ml import pasta_weight_app_v2/data/incoming_round2/labels.csv --destination pasta_weight_app_v2/data/experiment_002
.venv/bin/python -m pasta_weight_app_v2.pasta_ml validate pasta_weight_app_v2/data/experiment_002/dataset.csv
.venv/bin/python -m pasta_weight_app_v2.pasta_ml train pasta_weight_app_v2/data/experiment_002/dataset.csv --output pasta_weight_app_v2/data/experiment_002/model_v2.json
.venv/bin/python -m pasta_weight_app_v2.pasta_ml evaluate pasta_weight_app_v2/data/experiment_002/dataset.csv --model pasta_weight_app_v2/data/experiment_002/model_v2.json --split val --output pasta_weight_app_v2/data/experiment_002/model_val.json
```

この手順は指定済みsplitを取込時に保持するため、2回目の `split` コマンドは不要。検証は分割を跨ぐ同一グループを拒否する。新モデルでvalを確認した後、前節の `--split test --final-test` を新実験のCSV・モデル・出力パスで実行する。前のモデルを上書きせず、実験番号とモデル版ごとにデータハッシュ・結果・失敗画像を比較する。合成データのCLIテストで取込から再学習できることは確認できるが、追加写真で性能が改善したかは実写真の未使用評価で確かめる。
