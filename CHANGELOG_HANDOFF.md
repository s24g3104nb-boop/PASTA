# 引き継ぎ作成時の記録（2026-10-28）

- 元の `Pasta_Lens_Python.zip` を同梱コードの唯一の参照元とした。
- 元コードの `app.py`, `pasta_vision.py`, `storage.py`, `generate_demo.py`, `tests/`, `sample_images/`, `README.md`, `requirements.txt`, `start_windows.bat` は **内容を変更せず**収録。
- 2026-10-28 にテスト `pytest -q tests/test_core.py` を実行し **4 passed**。
- 実写真・AIモデル・実測データ・PC上のSQLite履歴は受領していない。
- 新たに作ったのは引き継ぎ資料・CSVテンプレート・.gitignoreのみ。学習済みモデルや精度の保証は含まない。
- エラーの履歴：StreamlitはユーザーWindows上で起動、Cloudflare Quick Tunnelは接続確立するがスマホで `Bad Request`。未解決。
