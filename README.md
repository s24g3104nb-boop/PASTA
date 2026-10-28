# Pasta Lens

乾燥スパゲッティの写真による重量推定の試作です。既存のStreamlit・OpenCV・製品登録・履歴機能を引き継ぎ、AI学習管理画面、写真取込・グループ分割・学習・評価CLI、材料入力による料理栄養計算を追加しています。

**実写真による学習・精度評価は未実施です。** 私有写真は検品と独立した撮影・計量・物理試料の対応確認を進めています。確定ラベルと候補重量を分離し、未承認画像や候補重量で学習しません。写真・受領CSV・マスクはGit対象外です。CGは実装テスト専用です。

## モバイルUI

ホーム／履歴／料理・栄養／設定の4画面に再構成しました。写真を確認してから解析し、結果を重量・栄養カードで表示します。学習管理と製品校正は設定から開きます。ライト／ダーク／OS追従表示に対応。詳しくは [画面構成と検証手順](docs/23_MOBILE_UI.md) を参照してください。

## 起動とテスト

リポジトリのルートで実行します。Python 3.12 / Linuxで確認しています。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-cloud.lock.txt
.venv/bin/python -m pytest -q pasta_weight_app_v2/tests
.venv/bin/python -m streamlit run pasta_weight_app_v2/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

`requirements-cloud.lock.txt` は今回のクラウド環境で検証した全依存の固定版です。依存更新は `pasta_weight_app_v2/requirements-dev.txt` の範囲を確認し、テスト後に固定版を更新してください。

Windowsのcmdでは次のように実行します（Windows実機は未検証）。

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r pasta_weight_app_v2\requirements-dev.txt
.venv\Scripts\python -m pytest -q pasta_weight_app_v2\tests
.venv\Scripts\python -m streamlit run pasta_weight_app_v2\app.py --server.address 127.0.0.1
```

画面は広げた麺・束の断面をOpenCVで数え、校正した1本重量と製品栄養値で計算します。手動本数の修正、製品の登録・更新、実測重量入力、画像と履歴の保存も使えます。AI学習管理で実写モデルを学習・選択すると、測定画面で実験用回帰へ切り替えられます。

認証は未実装です。現在はローカル利用に限定し、履歴・写真を無認証で外部公開しないでください。スマホ・Windows・Cloudflare経由の過去の400エラーはこの環境では再検証していません。

## 写真の取込から学習・評価まで

[乾麺の学習手順](docs/14_DRY_TRAINING.md) にCSV列、20〜100gの登録方法、実行コマンドをまとめています。

コードを編集せず写真の追加・編集・削除、別プロセスでのCPU学習、モデル選択、OpenCVとのval比較表・グラフを使うには [AI学習管理画面の手順](docs/17_TRAINING_MANAGEMENT.md) を参照してください。追加画像でも過去の評価分割を維持します。

重量未確認のパッケージは [受領・確認待ち・マスク検品](docs/18_PHOTO_INTAKE.md) を使います。受領だけでは学習台帳へ登録せず、撮影群・実測g・製品・撮影状態・画像QC・秤表示除外を確認します。[束の側面と断面](docs/19_CAPTURE_MODES.md) は区別します。

[ROI変換・背景除去・漏洩検証](docs/20_PREPROCESSING_QC.md) と [最新CSVの訂正・QC・学習前点検](docs/21_DATA_REVISION_AND_QC.md) を参照してください。バックアップ付きラベル訂正、原本/プレビューの来歴、物理試料を含む分割、HSV/Lab比較、スマホ用ブラシとUndo/Redoを利用できます。候補作成は承認や学習を自動実行しません。

- `validate` → `import` → `split` → OpenCV評価 → `train` → val評価 → 最終test評価 → `predict`
- 同一秤量・撮影セッション・同一／近似画像を同じグループに固定します。
- 束と広げた麺を製品別に扱うCPU画像特徴量＋リッジ回帰です。深層学習済みモデルではありません。
- 誤差指標と推定失敗率を併記します。未対応製品や訓練範囲外は棄却しますが、正しい推定を保証する判定ではありません。
- 実写真の検品と独立評価群が整わない状態では実写精度を算出しません。写真とCSVはGit対象外の `pasta_weight_app_v2/data/` に置いてください。

## 完成料理

[料理計算の仕様と使い方](docs/15_COOKING.md)。料理名は手動選択できます。料理画像AIは未学習です。

「料理と栄養」タブでは使った乾麺の履歴を選び、同じ調理バッチであることを確認します。材料・ソース・油・調味料の使用gと100g栄養値を入力して、乾麺分＋追加材料分＝料理合計を計算し、人数または食べる割合で按分します。利用者登録の材料テンプレートは概算と表示します。同じ乾麺履歴を別の料理へ二重保存する操作は拒否します。画像だけから材料量・正確なカロリーや完成料理の重量を推定しません。

## 保存データと移行

標準保存先は `pasta_weight_app_v2/data/` です。必要なら起動前に `PASTA_DATA_DIR` で別の保存先を指定できます。写真・DB・学習成果物をGitに登録しないでください。

既存のSQLiteに対して、計測の由来・推定方式・製品栄養値のスナップショット列と、料理用テーブルを追加します。既存の製品・計測・写真は削除しません。旧履歴の由来は `unknown` として警告します。自分のPCの既存データを移す前には、アプリを停止して `data/` 全体をバックアップしてください。

`README_FIRST.md`、`docs/01`〜`13`、`MANIFEST_SHA256.txt` は引き継ぎ時点の資料です。マニフェストは受領した元ファイルの照合用で、改修後コードのハッシュではありません。元コードは最初のコミットに保持しています。不足したCG2画像は既存生成器で再生成し、マニフェストとの一致を確認しました。
