# 方式B：前処理の安定性・再現性の研究

## 範囲

本番推定、UI、日記、SQLite、既存モデルは変更しない。前回の `direct_*` 実験も上書きせず、`stability_*` の追加モジュールで引き継ぐ。写真・モデル・台帳・レポートはgitignore対象の非公開 `data/` に保存する。

新規15原本を、名目重量での探索的当てはめにのみ利用する。5撮影群の物理的独立性は未確認。同重量3枚と全派生画像を同じ群に保持し、全重量・過去系列との関係も保守的に扱う。独立評価値は常に未算出。本番向け重量出力は `None`、状態は「要確認」。

## 追加モジュール

`pasta_weight_app_v2/experiments/` 以下：

| モジュール | 役割 |
|---|---|
| `stability_pipeline.py` | 共通の画像正規化・マスク・特徴量と、保存モデルの研究用推論アダプター |
| `stability_experiment.py` | 同一原本の変換、マスク・特徴量保存、前回特徴量の再現検査、形態学処理の比較 |
| `stability_models.py` | 同じ固定5モデルの比較、予測差・特徴変動・線形寄与のCSV |
| `stability_report.py` | 比較HTML・画像別ページ・グラフ |
| `stability_diagnostics.py` | 34.13g事例の反実仮想、マスク再生成、保存モデルによる一連の推論確認 |

新規単体テストは `tests/test_stability_pipeline.py`。既存テストと研究用 `experiments/research_tests/` も実行する。

## 共通処理と保存形式

`process(rgb, config)` を特徴量の学習入力生成と推定入力生成の両方で使う。外部から受け取る原本は既存 `image_to_rgb` でEXIFを補正し、RGB化・長辺最大1400pxへの縮小を行う。過去結果を再現するため、この1400px画像を変換試験の基準入力にする。

モデル成果物は `pipeline_version`, `pipeline_config`, `config_sha256`, `pipeline_source_sha256`, `feature_keys`, `model`, `production_eligible=False` を持つ。推論アダプターは設定ハッシュ・バージョン・特徴順の不一致を拒否する。実験JSONには画像処理と回帰それぞれの依存版も保存する。モデルのjoblibは信頼するローカル成果物だけを読み込む。

前処理の厳密再生は、特徴抽出時と同じNumPy/OpenCV版で行う。異なるNumPy版では、マスクが一致していてもPCA・幅プロファイルの浮動小数点境界に小差が出る場合がある。診断コードは依存版が異なる場合に拒否する。モデルの版だけでなく処理環境を固定する。

## 比較条件（結果を見た後の閾値探索なし）

| 条件 | 前処理 | 特徴 |
|---|---|---|
| A_hsv_legacy12 | 前回と同じHSV＋3×3 closing、入力解像度依存 | 前回の12特徴 |
| A_lab_legacy12 | 前回と同じHSV＋Lab、closingなし | 前回の12特徴 |
| B_normalized12 | 長辺512px、縦横比保持、HSV＋Lab | 前回の12特徴 |
| B_normalized5 | 同上 | 占有率、PCA投影の長短辺/画像長辺、密度、幅CV |
| C_clean5 | Bと同じ＋小領域/主領域から離れた短い領域を除外 | Bと同じ5特徴 |
| D_object4 | Cと同じ | 面積/対象長辺²、短辺/長辺、密度、幅CV |

縮小はINTER_AREA、拡大はINTER_CUBIC。極端な縦横比変更、隠れた量の補完、実寸の推定は行わない。Dは画像内の対象長辺を基準とする形状比の実験であり、トレーや麺の実寸が既知になった「物理スケール補正」ではない。重量の手掛かりも消す可能性を比較する。

PCA投影の長短辺は1–99百分位。幅プロファイルは主軸20区間、各区間の短軸5–95百分位の幅。端や孤立画素への感度を抑えるが、麺の全長や束の真の太さを測ったことにはならない。Cの除去率と候補マスクは必ず残し、正解マスクとして承認しない。

各原本に15条件：基準、50%縮小、25%縮小、その各復元、50%縮小時の線形/最近傍補間、JPEG品質85/60、±3度回転、明度0.9/1.1倍、90度回転、75%縮小して黒いキャンバス中央へ配置。回転は全原画像を残す拡張キャンバス。これにより背景割合も変わるため、向きだけの純粋な実験ではない。キャンバス試験も実カメラ距離の測定ではない。

225入力＝15原本の派生。6処理で1350特徴行、固定5モデルで6750研究予測。拡張画像は学習にも独立テストにも加えない。モデルは前回と同じ平均定数、線形、Ridge(alpha=10)、RandomForest(100木/深さ3/最小葉2)、GradientBoosting(50木/深さ1/率0.05)。新しい高容量モデルを増やしていない。

## 再実行

既存アプリ環境（画像処理）と研究用環境（scikit-learn等）を使う。実行時の正確な依存版は非公開成果物の `reproducibility.json` を正とする。画像前処理にはNumPy 2.5.4/OpenCV 4.14.0を使用。回帰段階にはNumPy 2.3.5/scikit-learn 1.8.0/joblib 1.5.3を使用した。これらを本番のrequirementsへ追加する必要はない。

```bash
# 作業前にデータと研究コードを別の非公開領域へバックアップする。
# 出力先は毎回新規ディレクトリ。前回の出力には上書きしない。
.venv/bin/python -m pasta_weight_app_v2.experiments.stability_experiment \
  --previous pasta_weight_app_v2/data/direct_weight_nov08 \
  --output pasta_weight_app_v2/data/stability_run02/extraction
python -m pasta_weight_app_v2.experiments.stability_models \
  --features pasta_weight_app_v2/data/stability_run02/extraction/features.json \
  --previous pasta_weight_app_v2/data/direct_weight_nov08 \
  --output pasta_weight_app_v2/data/stability_run02/regression
python -m pasta_weight_app_v2.experiments.stability_report \
  --root pasta_weight_app_v2/data/stability_run02
# diagnosticsには画像処理とsklearnの両依存が必要。
# NumPy/OpenCVは必ず特徴抽出と同じ版に固定する。
python -m pasta_weight_app_v2.experiments.stability_diagnostics \
  --root pasta_weight_app_v2/data/stability_run02
```

ここで `python` は各段階に対応する研究環境の実行ファイルに置き換える。今回の作業環境では、診断は画像用 `.venv/bin/python` に研究ライブラリのsite-packagesを追加して実行した。再現先では分離した研究venvに必要なライブラリを入れてよい。Windowsでは `bin/python` を `Scripts\python.exe` に置換し、コマンドを1行で実行できる。Windows実機での再現は未確認。

```bash
.venv/bin/python -m pytest -q pasta_weight_app_v2/tests
python -m pytest -q pasta_weight_app_v2/experiments/research_tests
```

`stability_report.py` は機械的な表・図を生成する。日本語の個別分析 `REPORT_JA.md` は目視検品とともに別途用意し、レポート内リンクを確認する。

## 結果を解釈するときの注意

- 訓練内MAE、同一原本からの予測変動、独立試料への誤差は異なる指標。自動テスト通過も重量精度を保証しない。
- 定数モデルの予測変動は0だが、重量を読み取っていない。安定性だけで最良モデルを決めない。
- 同一原本マスクとのIoUは自己一致度であり、正解マスクとの抽出精度ではない。回転・キャンバス変換は座標が変わるため自己IoUを未算出にする。
- 縮小・JPEG・光量変更で失われた細部は、拡大で復元できない。
- マスクQCは前景不足、画像端接触、低解像度、細部不足、大きな除去率の参考フラグ。手・別製品・同色の異物を完全には識別しない。
- 範囲外を20–100gにクリップせず研究値を記録する。未承認モデルの利用者向けgは返さない。

## 保護と復元

作業前に `data/` 全体と従来の `experiments/` を別の非公開tarに保存し、SHA-256を記録する。既存全ファイルのハッシュ台帳を保存し、終了時に変更・消失を確認する。バックアップ内容も台帳と照合する。DBマイグレーションは行わない。

復元が必要な場合はアプリを停止し、tarを**新しい空のフォルダ**に展開する。既存データへ直接展開しない。復元先 `data/` と保護台帳のハッシュを照合してから、アプリの保存先切替または復旧作業を行う。実行中SQLiteを上書きしない。個人写真・台帳を含むためバックアップもGitHubへ公開しない。

## 独立評価と次工程

独立した物理試料・親材料IDを確認してから分割を固定する。同じ麺、積み増し、再配分、連写、原本の派生は分割をまたがせない。不明な系列を別々の群と仮定しない。近似画像検索は共有関係の証拠の代わりにはならない。

初期案は5重量×非共有の5試料×3配置。重量ごとに3試料train/1validation/1testを事前固定し、距離・向き・背景・照明を重量と独立に割り当てる。この75枚だけで最終性能を証明できるという意味ではない。

暫定基準案：JPEG・縮小・軽い回転/明るさ変更の同一原本に対して95百分位差2g以内、最大差5g以内。これを通っても独立MAE10g以内（中間）、5g以内（最終）、最大誤差20g以内、出力率90%以上等を別に確認する。実距離変化や未知製品は別の試験であり、今回のデジタル変換試験の合格を流用しない。全て未達成/未確定の採用条件案である。
