# Pasta Lens

写真から乾麺の重さを確認し、食べたパスタを写真付きで残すStreamlitアプリです。クリーム・セージグリーンの共通UI、ライト／ダーク表示、カレンダーと写真付き食事日記を利用できます。

Windows（Python 3.12）では、このフォルダーを新しい場所へ展開し、`setup_windows.bat` → `start_windows.bat` をダブルクリックしてください。詳細・旧版からの安全な更新方法は [README_JA.md](README_JA.md)、スマートフォン接続は [SMARTPHONE_JA.md](packaging/windows/SMARTPHONE_JA.md) を参照してください。最新版のコードはGitHubのmainに統合します。

## 主な機能

- 写真を撮影・選択して乾麺を解析。未校正・検出失敗時は理由と次の操作を表示。
- 電子はかりによる実測重量を入力・保存。写真なしの実測記録も補助操作から利用可能。
- 製品の栄養情報がある場合の乾麺栄養計算、材料と分量に基づく料理の栄養計算。
- 料理写真・料理名・食べた日時・メモ、検索、カレンダー、詳細、編集・削除。
- 設定内の開発・検証ツール：A4撮影補助シート、マーカー検出、品質確認、未承認の研究用データ登録、既存学習管理。

写真による重量推定の実物精度は未検証です。校正済みの対応条件でも、推定値は電子はかりで確認してください。方式Bの研究モデルは独立試料での評価が未完了のため通常推定に採用していません。完成料理の写真からカロリーを推定するAIは準備中です。材料情報が不足する場合は「未計算」です。

## データと開発

写真・履歴は起動したPCのローカルに保存されます。Windows配布ZIPでは`user_data/`、GitHubのソース版と直接起動の既定では`pasta_weight_app_v2/data/`です。既存データを新しいフォルダーへ自動移動・統合しません。更新前にはアプリを終了し、保存フォルダー全体をバックアップしてください。DB形式の変更時には移行前バックアップを保存します。

本リポジトリには個人の写真・実測台帳・SQLite・学習済みモデルを含めません。2枚のサンプル画像はCGです。研究用画像は自動承認されず、同一物理試料・由来不明の試料のデータ漏洩を防ぐ管理を行います。

```bash
python -m venv .venv
# Windows: .venv\Scripts\python.exe
.venv/bin/python -m pip install -r pasta_weight_app_v2/requirements-dev.txt
.venv/bin/python -m pytest -q pasta_weight_app_v2/tests tools/tests
.venv/bin/python -m streamlit run pasta_weight_app_v2/app.py
```

方式Bの任意研究環境：`pasta_weight_app_v2/requirements-research.txt`。研究テストも含めるには `python -m pytest -q pasta_weight_app_v2/tests pasta_weight_app_v2/experiments/research_tests tools/tests`。通常利用には研究用依存ライブラリは不要です。

構成・設計資料は `docs/`、今回の統合・検証・復旧手順は `docs/37_INTEGRATION_RELEASE.md` に記載します。実際のiPhone Safari、Windows、印刷したA4用紙での確認はブラウザ自動検証と区別してください。

密集写真の重量推定は未解決です。[2026-11-12の診断・比較と追加撮影計画](docs/38_DENSE_WEIGHT_DIAGNOSIS.md)を参照してください。独立試料で未検証の回帰モデルは標準推定に採用していません。
