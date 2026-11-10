# 方式B：画像特徴量からの直接重量推定・研究環境

本番アプリと独立した受領検品・特徴量抽出・探索的回帰の手順。DB、UI、食事日記、製品校正、既存モデルの登録先は変更しない。画像・台帳・実験モデル・結果は非公開の `data/` 内に保存する。

## 評価上の制約

同重量の3画像は同じ撮影グループ。重量間・過去データとの物理試料の共有が未確認なら、全体を1つの保守的な系譜ブロックに保持する。5撮影群は5独立試料を意味しない。

ユーザーが明示した名目重量で探索的に当てはめることと、承認済み教師データへの昇格は別。今回のコードは前者だけを実施する。マスクは候補であり正解ではない。ラベルや承認フラグは更新しない。旧写真は重複確認だけに使用する。

正式な検証・テスト誤差は `null`。訓練内MAE/RMSE/MAPE/最大誤差を汎化性能と解釈しない。乱数による画像分割、ハイパーパラメータ探索、ベストモデルの自動採用は行わない。変換画像は学習枚数に加えず、同じ原本に対する感度確認にのみ使用する。

## 追加コード

| ファイル（`pasta_weight_app_v2/` 以下） | 役割 |
|---|---|
| `experiments/direct_intake.py` | ZIP検査、SHA-256・画素ハッシュ・回転対応pHash、系譜と承認状態の保持 |
| `experiments/direct_features.py` | 現行HSVとHSV+Lab、特徴量、縮小・回転感度、マスク比較HTML |
| `experiments/direct_models.py` | 固定パラメータ5モデル×3特徴集合×2マスク、訓練内指標、モデル保存 |
| `experiments/direct_report.py` | 特徴変動・訓練内比較・感度のグラフとHTML |
| `tests/test_direct_features.py` | 原本・既存データ保護、回転、QC、承認保留の検査 |
| `experiments/research_tests/test_direct_models.py` | 系譜分割拒否、再現性、探索モデルの出力制限・一連の処理 |

既存の研究コード `intake_receipt.py`、`diagnose_series.py`、`compare_masks.py` と本番の画像読み込み・検出関数を読み取り利用する。現行マスクの既定値は `developer_ui.py` の定数をASTで取得する。アプリ起動やDB接続は不要。

## 再現手順

リポジトリのルートで実行する。`data/direct_weight_run02` など毎回新しい出力先を指定する。上書きは拒否される。以下のZIPパスは手元の非公開ファイルに置き換える。

```bash
# 画像処理：既存アプリ環境
.venv/bin/python -m pasta_weight_app_v2.experiments.direct_intake \
  --zip /private/direct_weight_photos.zip \
  --existing pasta_weight_app_v2/data \
  --output pasta_weight_app_v2/data/direct_weight_run02
.venv/bin/python -m pasta_weight_app_v2.experiments.direct_features \
  --receipt pasta_weight_app_v2/data/direct_weight_run02 \
  --output pasta_weight_app_v2/data/direct_weight_run02/features

# 回帰：本番の依存に追加せず、研究用venvを用意する
python -m venv /tmp/pasta-direct-research
/tmp/pasta-direct-research/bin/python -m pip install \
  numpy==2.3.5 scikit-learn==1.8.0 scipy==1.17.0 \
  matplotlib==3.10.8 joblib==1.5.3 pytest
/tmp/pasta-direct-research/bin/python pasta_weight_app_v2/experiments/direct_models.py \
  --input pasta_weight_app_v2/data/direct_weight_run02/features/features.json \
  --output pasta_weight_app_v2/data/direct_weight_run02/models_experiment
/tmp/pasta-direct-research/bin/python pasta_weight_app_v2/experiments/direct_report.py \
  --root pasta_weight_app_v2/data/direct_weight_run02
```

Windowsでは研究用venvを任意のローカルフォルダに作り、各 `bin/python` を `Scripts\python.exe` に置き換える。シェルの行継続を使わず1コマンド1行でもよい。本番起動方法は変更しない。今回Windows実機での研究コマンド実行は未検証。

今回実行時のPython・OpenCV・NumPy・scikit-learn等の版、ソースハッシュ、入力ハッシュ、乱数、パラメータは非公開結果JSONと `reproducibility.json` に保存する。画像処理と回帰のNumPy版が異なる環境で実行したため、それぞれの版を分離して記録する。

`analysis/index.html` は機械的な比較結果。`REPORT_JA.md` は目視所見と判断を含む個別の日本語報告として別途作成する。再実行時もその報告を作成してからHTML内の報告リンクを確認する。

## 特徴量と比較条件

- 元画像はEXIF補正後、長辺最大1400px。比較用に半分縮小と90度回転を追加する。デジタル縮小は実際の撮影距離の再現ではない。
- `compact3`：占有率、画像長辺で正規化した外接矩形の長辺・短辺。
- `shape12`：上記、縦横比、外接矩形内密度、正規化輪郭長、相対幅中央値、幅の変動、512px基準の骨格密度、Lab色、明度分散。
- `background_control`：背景明度・ばらつき・白飛び割合、向き、画像縦横比。背景・構図がラベルと結び付く交絡の診断専用。
- 別途20区間の主軸直交幅プロファイル、面積、連結成分数、骨格長等をJSON/CSVに保存。面積・幅・骨格はマスクの特徴であって物理寸法・本数の正解ではない。
- 平均定数、標準化した線形回帰、Ridge（alpha=10）、RandomForest（100木・深さ3・最小葉2）、GradientBoosting（50木・深さ1・学習率0.05）。乱数42、探索なし。
- 同じ15原本で全候補を比較。shape12には撮影グループ単位のラベル入れ替え10種も適用し、訓練データを当てはめる能力を診断する。有意差検定や正式な外部検証ではない。

`models/*.joblib` は未承認の研究用成果物。信頼できる作成元のファイルだけを読み込む。本番モデル管理からは参照しない。公開向け `public_result()` は常に「要確認」、重量は `None`。信頼区間・信頼度百分率は捏造しない。

## テストと保護確認

```bash
.venv/bin/python -m pytest -q pasta_weight_app_v2/tests
/tmp/pasta-direct-research/bin/python -m pytest -q pasta_weight_app_v2/experiments/research_tests
.venv/bin/python -c "from pasta_weight_app_v2.experiments.intake_receipt import verify_protection; print(verify_protection('pasta_weight_app_v2/data/direct_weight_run02'))"
git diff --check
git diff --stat
```

個人データを使わない合成画像・一時ディレクトリのテスト。既存データは受領直前の全ファイルSHA-256と照合し、削除・変更を検査する。原本はlabels.csvとsha256.txtの両方と照合する。プレビュー・変換画像は別パスと別ハッシュを持つ派生物であり、原本の重複として登録しない。

## 次のデータ台帳と本番採用条件

台帳には `image_sha256, original_id, physical_family_id, sample_id, parent_material_ids, capture_group_id, exact_measured_g, scale_resolution_g, product_id, diameter_mm, length_mm, camera_id, distance_cm, angle_deg, lighting, background_id, arrangement, mask_review, label_review, split` を持たせる。分からない値は不明のまま保持する。

同じ麺・積み増し・再配分・原本の派生物は分割をまたがせない。写真が異なることは物理試料が異なる証拠にならない。コードの分割ガードは系譜確認フラグと明示分割を要求するが、そのフラグを正当化する計測・人的確認は別途必要。

最初の追加試験案は、5重量×互いに麺を共有しない5試料×配置3通り=75枚。各重量で3試料を学習、1試料を検証、1試料を封印テストに固定する。これは実験開始の最小構成案であり、高精度を証明する十分な標本数ではない。物理共有があれば家族単位で組み直す。距離・背景・向きを重量に偏らせず割り当てる。

本番採用には、正解重量・マスクQC・系譜の確認、独立試料でのMAE/RMSE/最大誤差/重量別・条件別誤差/拒否率、縮小・回転への安定性、未知製品と範囲外入力の拒否、既存方式との同一評価条件での比較が必要。MAE10g以内は中間目標、5g以内は最終目標。今回の訓練内指標で達成判定しない。
