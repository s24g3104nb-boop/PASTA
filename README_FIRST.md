# Pasta Lens — Codex引き継ぎパッケージ

**基準日：2026-10-28 / 言語：日本語 / 対象：Codex（コード実装・検証）**

このZIPは、ユーザーとChatGPTが開発してきた乾燥パスタ重量推定Webアプリの **既存ソースコード＋引き継ぎ文書** のセットです。
「AI学習済みのアプリ」ではありません。**現状はOpenCVで本数を検出する試作版で、実物写真による検証も未実施**です。

## Codexへの渡し方
1. このZIPを解凍し、**このフォルダ全体（`pasta_weight_app_v2/` と `docs/` を含む）**をCodexの作業フォルダ／PASTAリポジトリに配置する。**文書だけ7個を添付してもコードは渡らない**。
2. Codexに `CODEX_START_PROMPT.md` と `CODEX_UPDATED_INSTRUCTIONS.md` の全文を貼り付ける。
3. 既存コードは `pasta_weight_app_v2/` にある。必ず実コードを読んでから変更する。
4. 最初に `python -m pytest -q pasta_weight_app_v2/tests/test_core.py` を試す。依存がなければ仮想環境で `requirements.txt` をインストールする。
5. まず **現状の再現・テスト・課題洗い出し・改善の実装** を行う。写真収集は後から。

## 主要ファイル

| パス | 内容 |
|---|---|
| `CODEX_START_PROMPT.md` | 最初にCodexへ送る指示文。まずこれを読む |
| `AGENTS.md` | Codexの実装方針・制約・成果物ルール |
| `pasta_weight_app_v2/` | **以前作成した既存Pythonソース一式（無改変）** |
| `docs/01_PROJECT_BRIEF.md` | 背景・目標・現在地 |
| `docs/02_REQUIREMENTS.md` | 機能／非機能要件・優先順位・受入条件 |
| `docs/03_AS_IS_ARCHITECTURE.md` | 既存コードの構成と技術・DB |
| `docs/04_ML_CV_PLAN.md` | CV・機械学習の開発・評価方針 |
| `docs/05_DATA_COLLECTION_PROTOCOL.md` | 実測データの収集方法 |
| `docs/06_TEST_PLAN.md` | 自動／手動テスト、受入基準 |
| `docs/07_DEPLOYMENT_TROUBLESHOOTING.md` | Windows起動、スマホ公開、過去のエラー |
| `docs/08_ROADMAP_AND_TASKS.md` | 開発タスクを順序立てて整理 |
| `docs/09_DECISIONS_OPEN_ITEMS.md` | 確定済み要件と未確定事項 |
| `docs/10_UI_FLOW.md` | 画面・利用シーン・導線 |
| `docs/11_RISKS_LIMITATIONS.md` | 重量推定の原理的な限界・安全性 |
| `docs/12_REFERENCE_LINKS.md` | 公式資料リンク |
| `docs/13_COOKED_DISH_AND_CALORIES.md` | 完成料理の種類判別＋材料ベース栄養計算の更新方針 |
| `CODEX_UPDATED_INSTRUCTIONS.md` | 今回判明した写真データ・料理拡張の追加指示 |
| `dataset/dataset_template.csv` | データセット管理用の列定義（実データは入っていない） |
| `ORIGINAL_SOURCE_SHA256.txt` | 元のZIPファイルのハッシュ |

## 再現できた事実（2026-10-28時点）
- 元ソース：ユーザーとの以前の会話で生成した `Pasta_Lens_Python.zip` を展開して収録。
- `pasta_weight_app_v2/tests/test_core.py` は **4テストPASS**（合成画像2種類、栄養計算、SQLite保存）。
- **この環境ではStreamlit画面の実起動テストはしていない**（`streamlit` パッケージ未導入）。ただしユーザー本人のWindows端末ではStreamlitの起動成功画面を確認済み。
- ユーザー本人のWindowsでは `py` コマンドが使えず、`python -m streamlit run ...` で起動成功した。
- Cloudflare Quick Tunnelは接続ログが正常だったが、スマホから **Bad Request** が発生。原因未特定／未解決。
- ユーザーは **乾麺20g・40g・60g・80g・100gの実写写真を保有**していると報告した。ただし写真・ラベルCSV・実重量データの現物は**このZIPには含まれない**。枚数・撮影状態・製品・測定方法・品質は未確認。学習済みモデルも含まれない。

## 注意
- `data/`（実行時に生成される写真・SQLite DB）はソースZIPに含まれていない。ユーザーPCの実データも取得していない。
- Cloudflare一時URL、PCのIP、個人のメールアドレスなど、実行環境・本人情報は引継資料に埋め込まない。
- GitやCodexへの共有時は `data/`, `datasets/raw/`, `*.sqlite3`, `.env`, 認証情報をコミットしない。
- ユーザーはすでに乾麺の複数重量の写真を保有。最初に学習・評価の受け皿を実装し、写真が作業環境に取り込まれたら実データの検証へ移ること。**写真をまだ添付されていない段階で学習済みと主張しない**こと。
