# Pasta Lens

写真から乾麺の重さを確認し、食べたパスタを写真付きで残すStreamlitアプリです。クリーム・セージグリーンの共通UI、ライト／ダーク表示、カレンダーと写真付き食事日記を利用できます。

Windows（Python 3.12）では、このフォルダーを新しい場所へ展開し、`setup_windows.bat` → `start_windows.bat` をダブルクリックしてください。詳細・旧版からの安全な更新方法は [README_JA.md](README_JA.md)、スマートフォン接続は [SMARTPHONE_JA.md](packaging/windows/SMARTPHONE_JA.md) を参照してください。最新版のコードはGitHubのmainに統合します。

## 主な機能

- 撮影・選択した乾麺の写真を自動解析。条件が合うと、製品登録なしで乾麺の面積から概算重量を表示。
- 電子はかりによる実測重量を入力・保存。写真なしの実測記録も補助操作から利用可能。
- 乾麺は製品の栄養情報または未登録時の標準成分値で計算。完成料理は材料と分量から計算。
- 料理写真・料理名・食べた日時・メモ、検索、カレンダー、詳細、編集・削除。
- 「記録」→「料理・栄養」で、保存した乾麺と材料から食べる分の栄養を計算。製品の登録と校正は管理者アプリの設定で行えます。
- 基準シートを使わない乾麺撮影と、完成料理の候補から食事記録へ進む操作。完成料理の端末内AIは追加設定後に使えます。

面積の換算には20〜50gの写真を使用しています。独立した試料での精度は未検証です。推定値は電子はかりで確認し、実測値に修正できます。完成料理のカロリーは保存した乾麺の重量と材料・分量から計算します。

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

密集写真の重量推定は未解決です。 [2026-11-12の診断・比較と追加撮影計画](docs/38_DENSE_WEIGHT_DIAGNOSIS.md)

製品の新規登録・校正更新後の重量表示と写真診断の修正は [2026-11-18の報告](docs/39_WEIGHT_DISPLAY_FIX.md) を参照してください。

通常の撮影条件と追加の検証計画は [基準シートなしの重量推定](docs/44_SHEETLESS_WEIGHT.md) を参照してください。A4による計測や以前のパイロットは管理者向けの比較実験です。

[端末内画像認識・管理者アプリ・評価手順](docs/43_LOCAL_VISION.md)を参照してください。
