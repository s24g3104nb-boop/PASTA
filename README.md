# Pasta Lens

写真から乾麺の重さを確認し、食べたパスタを写真付きで残すStreamlitアプリです。クリーム・セージグリーンの共通UI、ライト／ダーク表示、カレンダーと写真付き食事日記を利用できます。

Windows（Python 3.12）では、このフォルダーを新しい場所へ展開し、`setup_windows.bat` → `start_windows.bat` をダブルクリックしてください。詳細・旧版からの安全な更新方法は [README_JA.md](README_JA.md)、スマートフォン接続は [SMARTPHONE_JA.md](packaging/windows/SMARTPHONE_JA.md) を参照してください。最新版のコードはGitHubのmainに統合します。

## 主な機能

- 撮影・選択した乾麺の写真を自動解析。写っている麺の面積から計算できる場合は、製品登録なしで概算重量を表示。
- 電子はかりによる実測重量を入力・保存。写真なしの実測記録も補助操作から利用可能。
- 製品の栄養情報がある場合の乾麺栄養計算、材料と分量に基づく料理の栄養計算。
- 料理写真・料理名・食べた日時・メモ、検索、カレンダー、詳細、編集・削除。
- 基準シートを使わない乾麺撮影。麺を平らに広げ、両端が入るよう真上から撮ります。研究用の管理者アプリと比較実験は[基準シートなしの重量推定](docs/44_SHEETLESS_WEIGHT.md)を参照してください。

面積の換算には20〜50gの写真を使用しています。見切れの可能性、麺の密集、換算範囲外は概算とあわせて確認できます。写真の外や重なりで隠れた麺の量は分からず、独立した試料での精度は未検証です。記録する概算重量は編集でき、電子はかりの実測値は別に入力できます。完成料理のカロリーは保存した乾麺の重量と材料・分量から計算します。

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

## 詳しい仕様・研究資料

- [食事日記と設計変更](docs/25_MEAL_DIARY.md)
- [手動重量入力と保存・移行](docs/36_MANUAL_WEIGHT_RECOVERY.md)
- [UI改修](docs/35_UI_POLISH_REPORT.md)
- [独立試料の撮影・登録計画](docs/32_CAPTURE_COLLECTION_PLAN.md)
- [方式Bと安定性診断](docs/30_DIRECT_WEIGHT_STABILITY.md)
- [データ漏洩を防ぐ分割](docs/31_INDEPENDENT_CAPTURE_PROTOCOL.md)

過去の受領資料・`MANIFEST_SHA256.txt`は当時の資料で、現在のコードの照合値ではありません。配布ZIPには最新版用のハッシュ台帳を同梱します。研究用の訓練内誤差を未知試料の精度として扱いません。
