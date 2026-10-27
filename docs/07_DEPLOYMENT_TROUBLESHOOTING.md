# 07 起動・スマホ利用・既知の接続問題

追記：現在の実装のローカル手順はリポジトリREADMEを参照。今回LinuxではStreamlit起動・HTTP 200・ヘルス応答とAppTestを確認した。スマホ、トンネルと過去の400は未再現。以下の過去記録にある `--allowed-mail` は、実行したツールと版が不明のため、現在のcloudflaredの正式オプションや400の原因と断定しない。

## ユーザー環境で観測できた事実
- Windows PCで `start_windows.bat` を実行。Python 3.11環境で必要ライブラリのインストールが成功し、Streamlitが起動。
- エクスプローラーの `app.py` のあるフォルダでアドレス欄に `cmd` と入力してcmdを開く運用が可能。
- `py ...` は **`'py' は ... 認識されていません`** とエラー。`python ...` に変更すると起動成功。
- `python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501` でサーバーが起動した（ポート8501）。
- Streamlitは `use_container_width` の非推奨警告を表示（アプリ停止の原因ではない）。
- 学内ネットワークのWi-Fiからスマホ→PCのLANアクセスに失敗。端末間通信制限の可能性はあるが確定していない。
- Cloudflare Quick Tunnelは **DNS、UDP/TCP、API接続チェックPASS、トンネル接続確立** のログを出した。
- `--allowed-mail` 付きトンネルからスマホへ接続したところ、ブラウザに **`Bad Request`**。原因は**未特定**。認証サービス、URL、リバースプロキシヘッダー、Streamlit側、スマホ側を切り分ける必要あり。

## ローカル起動手順（Windows、cmdで作業）
```bat
cd /d "<解凍したフォルダ>\pasta_weight_app_v2"
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
PCブラウザ：`http://localhost:8501`。**CMDを閉じるとサーバーが停止**。

## ローカルネットワーク接続
- 信頼できる同一ネットワークなら、PCで `ipconfig` によりその時点のIPv4を確認。
- スマホから `http://<PCのLAN-IP>:8501`。
- 学内・公衆Wi-Fiの端末間制限、ファイアウォール、VPNなどを調べる。制限回避のための非許可操作はしない。
- ブラウザのカメラは安全なオリジン（通常HTTPS）が必要なことがある。まずは画像アップロードで検証。

## Cloudflare Quick Tunnel（テスト目的のみ）
公式ドキュメント： https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/

- 認証なしQuick Tunnelは**URLを知る人がアクセスできる**。現行アプリにはログイン・ユーザーデータ分離がないため、**ユーザーの写真や履歴データを保存した状態で認証なし公開しない**。
- メール認証はCloudflareの `--allowed-mail` で設定できるが、今回の400エラーの原因は不明。
- 公開前に同じ端末で `http://localhost:8501` が開けることを確認し、トンネルのURLへPCとスマホでアクセスし、認証画面に遷移するかを比較。
- エラー再現では **時刻・HTTPステータス・Streamlitのログ・cloudflaredのログ（秘密情報除外）**を記録。
- ブラウザのシークレットモード／キャッシュ／端末違いで挙動の差を調べる。安全な検証環境で、認証サービスが原因かを段階的に切り分ける。
- 調査時は一時URLやメール、トークンを公開Issueやスクショに載せない。
- Quick Tunnelは試験専用でURLは都度変わる。常時公開は別のホスティング・認証・永続ストレージが必要。

## 将来のクラウド運用
- Streamlit Community Cloud等ではローカルSQLiteやアップロード写真の永続保存が保証されない環境がある。永続DB・オブジェクトストレージ、認証、バックアップを別途設計。
- 「PCの電源を切っても使える」はクラウドへデプロイした場合のみ。Tunnel単体ではPCが必要。

## 400を再調査するときの順序

1. 実データを含まない別の `PASTA_DATA_DIR` を用意し、READMEのローカル起動コマンドでPCから確認する。`python -m streamlit version`、使用したトンネルプログラムの名前・版・秘密情報を除いた引数を記録する。
2. PC内でルート画面と `/_stcore/health` が正常か確認する。前者は画面の土台、後者はサーバー生存の確認で、WebSocketを含む操作成功とは分けて記録する。
3. 許可された環境でPCブラウザとスマホの同じ経路を比較し、どの段階で400になるかを記録する。画面HTML取得、認証遷移、WebSocket接続、画像アップロードを分け、時刻とHTTPステータスをStreamlit側・トンネル側ログと照合する。トークン、Cookie、個人メール、一時公開URLは共有ログから除く。
4. ローカルでも失敗するならアプリ/Streamlitを、ローカル正常で中継時だけ失敗するなら接続URL、認証、Host/Origin、WebSocketの中継を順に調べる。観測なしに原因を決めず、CORS/XSRF/TLS検証の無効化を解決策としない。

カメラが使えない場合はスマホ標準カメラで撮影してアップロードできる。HTTPSでのカメラ許可・画面幅・接続は実機確認が必要。認証のない現行アプリを実データ入りで一般公開しない。
