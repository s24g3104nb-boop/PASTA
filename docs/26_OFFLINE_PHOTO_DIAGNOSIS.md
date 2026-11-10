# 非公開の写真系列を受領・診断する手順

写真の受領と、学習への登録・承認は別です。この手順ではUI、推定器、既存SQLite、学習データ、モデルを変更しません。実写真・付属CSV・生成マスク・診断結果は公開対象外の `data/` に保存します。

アプリや他のデータ更新処理を停止した状態で実行してください。既存ファイルのハッシュを前後で比較するため、実行中に別の書込みがあると不変性を確認できません。

```sh
# 新しい出力先を指定。既存フォルダーへの上書きは禁止。
.venv/bin/python -m pasta_weight_app_v2.experiments.intake_receipt \
  --zip PRIVATE_ARCHIVE.zip \
  --existing pasta_weight_app_v2/data \
  --output pasta_weight_app_v2/data/NEW_RECEIPT

.venv/bin/python -m pasta_weight_app_v2.experiments.diagnose_series \
  --originals pasta_weight_app_v2/data/NEW_RECEIPT/originals \
  --output pasta_weight_app_v2/data/NEW_RECEIPT/diagnostics
```

Windowsでは `.venv/bin/python` を `.venv\Scripts\python` に置き換え、各コマンドを一行で実行できます。

受領器は名目20/40/60/80/100g×3モード、全15枚・単一物理試料系列の管理形式に対応します。README、labels.csv、SHA256SUMS、画像の完全性・寸法・EXIF向きを照合します。ZIP内のパス逸脱・リンク・暗号化・不正ファイル・サイズ超過・上書きを拒否します。原本は無変換保存し、既存ファイルとのSHA-256完全重複を記録します。再圧縮や切抜きの近似重複は別途検品が必要です。

`receipt.json` と `protection_before.json` は個人データです。`verify_protection(output)` で既存ファイルの変更・欠落を検査できます。既存台帳には自動登録しません。名目重量の利用者確認、正確な小数実測値、撮影条件、製品校正、QC、学習承認を混同しません。未確認項目は空欄のままとし、全系列は分割未割当・学習対象外です。

診断器はアプリ画面の設定を `developer_ui.py` の定数から読み取り、既存検出器と同じEXIF補正・最長辺1400pxで処理します。断面の画面既定と検出関数の既定値を別記録します。生成物は次のとおりです。

- `index.html`：画像別の可視化一覧。
- 画像別ディレクトリ：入力、検出、マスク、連結成分、輪郭、骨格、closing前のマスク。
- `diagnostics.json`：原本・ソースハッシュ、基準コミット、依存バージョン、パラメータ、特徴量、縮小・回転への反応。
- `features.csv`：面積・領域数・輪郭・長短軸・密度・512pxでの骨格特徴。

検出候補数は正解本数ではありません。背景や手を含むマスクは未承認です。骨格長や幅は真の麺長・麺径ではありません。デジタル縮小は実際の撮影距離・遠近差を再現した実験ではありません。1系列内の重量との相関は記述統計に限定し、独立精度・学習成果とは呼びません。重量予測・MAE等は出力せず、モデル学習や自動採用も行いません。

```sh
.venv/bin/python -m pytest -q pasta_weight_app_v2/tests
.venv/bin/python tools/check_public_tree.py --history
```

テストは合成画像・一時フォルダーを使用します。公開時はコードと合成テストだけを選び、生成写真・CSV・JSON・原本・DBを追加しないでください。実際の画像ごとの診断結果は私用受領フォルダー内の報告書を参照します。
