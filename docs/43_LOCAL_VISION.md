# 端末内画像認識と管理者アプリ

[基準シートなしの比較・追加撮影計画](44_SHEETLESS_WEIGHT.md)

## iPhoneから記録する

1. Windowsで `setup_windows.bat` を実行し、通常は `start_windows.bat` で起動します。iPhoneからは同じ信頼できるWi-Fi上で `start_lan.bat` を起動し、SafariでPCのIPv4アドレスとポート8501を開きます。
2. 家庭のHTTP接続では、iPhoneの標準カメラで撮って「乾麺の重さをはかる」から画像を選びます。ブラウザ内カメラにはHTTPSかその端末のlocalhostとカメラ権限が必要です。
3. 一層に広げ、全体が見えるよう真上から撮ります。製品を選び、写真を解析します。重量が出ない場合は、電子はかりで測った数値を入力します。
4. 乾麺を保存したら「この乾麺で料理を記録する」を開きます。できあがった料理の写真を選び、候補を確認したうえで料理名を入力します。「材料を入力して栄養を計算する」で使った食材と量、食べた割合を選びます。標準レシピから仮入力した場合は概算で、食材と量を修正できます。
5. 栄養を確認し、元の食事記録に戻って保存します。記録とカレンダーから表示・編集・削除できます。1件の乾麺記録を別の食事にもう一度関連付ける操作は保存時に止まります。複数人分は料理の取り分け設定で指定します。

[家庭内LANの手順](../packaging/windows/SMARTPHONE_JA.md)

## Windowsで端末内AIを追加

CPU版から開始できます。管理者はPowerShellを開き、通常のセットアップで作成した専用仮想環境へ追加でインストールします。GitHubのコードと個人データのあるフォルダーは分けて管理します。

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
.\.venv\Scripts\python.exe -m pip install -r pasta_weight_app_v2/requirements-local-vision.txt
.\.venv\Scripts\python.exe -m pip install "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2"
```

次の公式ファイルを自分のPCの非公開フォルダーへ保存します。写真による推論はPC内で実行します。任意のCUDA対応GPUを使うには、CPU版の代わりに対応するPyTorch・torchvisionを用意し、`PASTA_LOCAL_VISION_DEVICE=cuda` を設定します。

| 用途 | モデル | 取得元と条件 |
|---|---|---|
| 乾麺の抽出 | MobileSAM ViT-T | 公式の [MobileSAM重み](https://github.com/ChaoningZhang/MobileSAM/blob/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt) 約41 MB。[Apache 2.0](https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE) |
| 完成料理の候補 | OpenCLIP ViT-B/32 QuickGELU LAION400M E32 | 公式 [OpenCLIP重み](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt) 約605 MB。[ソースライセンスと配布情報](https://github.com/mlfoundations/open_clip) |

```powershell
$env:PASTA_MOBILESAM_CHECKPOINT = "C:\PastaLensModels\mobile_sam.pt"
$env:PASTA_OPENCLIP_CHECKPOINT = "C:\PastaLensModels\vit_b_32-quickgelu-laion400m_e32-46683a32.pt"
.\start_windows.bat
```

iPhoneで利用するときは、同じPowerShellで `.\start_lan.bat` を起動します。環境変数はこのPowerShellの子プロセスに引き継がれます。AIの設定がない場合は、従来の重量推定、実測記録、料理名の選択、栄養計算を利用できます。測定に使ったモデルはGitHub・配布ZIPへ同梱しません。Windows実機の導入とiPhoneのカメラ・ネットワークは別途確認が必要です。

## 管理者の起動方法

管理者アプリには通常版と重ならない専用フォルダーを指定します。今ある研究用フォルダーを使う場合は、正常なバックアップから別の場所へコピーし、通常データや旧ファイルを上書きせずに指定します。通常版から管理者への画面遷移はありません。

```powershell
$env:PASTA_ADMIN_DATA_DIR = "C:\PastaLensAdmin\admin_data"
$secret = Read-Host "管理用パスワードを入力（12文字以上）" -AsSecureString
$env:PASTA_ADMIN_PASSWORD = [System.Net.NetworkCredential]::new('', $secret).Password
.\.venv\Scripts\python.exe -m streamlit run pasta_weight_app_v2/admin_app.py --server.address=127.0.0.1 --server.port=8502
```

管理者画面は自分のPCの `http://127.0.0.1:8502` でパスワードを入力して利用します。LAN待ち受けで起動した場合はログイン画面を出す前に拒否します。管理者プロセスを公開プロキシやトンネルの接続先に指定しないでください。通常と管理者のSQLite・写真・研究領域・ログは別です。必要な場合は `PASTA_ADMIN_RESEARCH_DIR` で管理者の研究領域を指定できます。

## 管理者向けのA4計測・校正

MobileSAMはHSV＋Labの候補から作った枠を使って可視領域を抽出します。960pxの入力を使い、4つの基準マーカーから射影変換で紙面座標へ移します。定規で100mmの印刷線を確認し、紙面に一層で置いた場合に、投影面積・長さ・幅が求まります。反射、はみ出し、マーカーの欠落、層が重なった配置は撮り直す条件です。管理者はマスクを見比べ、背景の混入や麺の欠けを確認してから、投影面積を校正に使います。

Aの換算は製品ごとの `g = a × 面積(mm²) + b` です。係数は実測と一層配置を確認した写真から求めます。開発用の少数の確認では、別々の材料を使う校正3試料・確認2試料を最低としました。暫定条件は確認MAE 5g以下、最大誤差10g以下、換算は校正範囲内です。これを通っても表示は「検証中」です。管理者画面の面積計測でマスクを確認し、面積記録JSONを取得します。各記録に `physical_sample_id`、`measured_g`、`disjoint_material_confirmed`、事前に決めた `split`（`fit` または `check`）を加え、`{product_name, backend, samples}` の形にまとめます。

```powershell
.\.venv\Scripts\python.exe -m pasta_weight_app_v2.experiments.area_calibration --samples private_samples.json --output area_profile.json
$env:PASTA_AREA_CALIBRATION = "C:\PastaLensModels\area_profile.json"
```

3つの目標重量は20・50・80gを目安にします。まず同じ製品の20・50gを各1試料、マーカーを隠さず全体が一層で収まるか確認します。収まればこの条件を固定し、校正用は20・50・80gの3試料、別の確認用は30・65gの2試料を追加します。可能なら各試料で距離だけ変えて1枚補足し、同一試料の別撮影として扱います。80gが収まらなければ上限を下げます。100〜200gはA3以上の平らな板に既知の間隔でマーカーを置く設計で再検証します。A4のPDFを単純拡大してA4用の推定へ流用すると、投影寸法がずれます。

## 管理者向け比較の実行

`foundation_benchmark segment` は指定した検出器を単独のプロセスで実行し、マスク、重ね合わせ、注釈付き写真のIoU・混入・見逃し、紙面補正の状況、処理時間とピークメモリーを出力します。`tools/foundation_mask_editor.html` をブラウザーで直接開くと、原本の向き・SHA-256を確認し、線や多角形で注釈を作成・修正できます。各実行は新規フォルダーに出力し、元写真・来歴のファイルは保存し直しません。

```text
python -m pasta_weight_app_v2.experiments.foundation_benchmark segment --dataset ORIGINALS --dataset OTHER_ORIGINALS --lineage-revision CUMULATIVE.json --backend opencv_hsv --annotations ANNOTATIONS --output PRIVATE_HSV
python -m pasta_weight_app_v2.experiments.foundation_benchmark segment --dataset ORIGINALS --dataset OTHER_ORIGINALS --lineage-revision CUMULATIVE.json --backend mobile_sam --checkpoint mobile_sam.pt --annotations ANNOTATIONS --output PRIVATE_SAM
python -m pasta_weight_app_v2.experiments.foundation_benchmark regress --dataset ORIGINALS --dataset OTHER_ORIGINALS --lineage-revision CUMULATIVE.json --segmentation PRIVATE_HSV --segmentation PRIVATE_SAM --output PRIVATE_FITS
```

- A4投影がある写真ではAの面積のみとBの面積＋長さ・幅・形状を同じ対象で比較します。`segment --a4-reviews` に、原本SHA-256をキーにした `printed_line_mm` と `flat_sheet_confirmed` のJSONを指定します。重い10枚は同じ材料を加えた1系列です。24枚は独立テストに使いません。
- Cの物理近似は面積を1本の長さ×直径で割り、1本の実測重量を掛けます。`regress --product-geometry` に同じ原本SHA-256をキーにした `measured_product_confirmed`、`strand_length_mm`、`strand_diameter_mm`、`grams_per_strand` を渡します。重なりがある場合に見える面積は減ります。
- DINOv3を使うDでは、`--dinov3` にローカルのViT-S/16 safetensorsスナップショットを渡します。固定エンコーダーの384次元を全体写真、または背景を灰色に置換した乾麺領域から取得します。回帰は標準化と固定Ridgeです。背景色だけの負の対照を同時に計算します。
- `foundation_evaluation --features FEATURES.json --plan SPLIT.json --partition test --output EVAL.json` は来歴でグループを検証し、あらかじめ指定した `feature_variant` のみを学習用試料で当てはめます。[分割の形式と指標](41_SINGLE_PHOTO_NEXT.md)の `records`・`assignments`・`material_parents`・`target_weights_g` に加え、テストでは `frozen_before_test: true` の確認を記録します。停止を分母に入れ、MAE・RMSE・最大誤差を返します。

| 研究候補 | 使用条件 |
|---|---|
| SAM 2.1 Tiny | 38.9Mパラメーター。Meta公式の `sam2.1_hiera_tiny.pt` とSAM 2を使用。CPU経路は合成の初期化重みで確認済み。WindowsではMetaがWSLを推奨。PyTorchは2.5.1以上、torchvisionは0.20.1以上。[公式コード・Apache 2.0](https://github.com/facebookresearch/sam2) |
| SAM 3.1 | `--backend sam31 --device cuda`。MetaのObject Multiplexを1フレーム・固定テキストで使用。Python 3.12以上、PyTorch 2.7以上、CUDA 12.6以上が公式要件。VRAMと乾麺の精度は未計測。[公式リリース](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)・[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE) |
| DINOv3 ViT-S/16 | Transformersを別途追加（接続試験では4.57.6）。Meta提供重みへのアクセス申請・同意後、ローカルから読み込みます。研究結果公表時の謝辞と再配布時の条件を確認します。[公式配布・ライセンス](https://github.com/facebookresearch/dinov3) |

料理の標準レシピは乾麺100gを基準にした分量例です。栄養値は [USDA National Nutrient Database SR28](https://www.ars.usda.gov/ARSUserFiles/80400525/Data/SR/SR28/sr28_doc.pdf) の項目を [公開SR28データのミラー](https://github.com/alyssaq/usda-sqlite/tree/master/data) で食品番号と照合しました。使用量、商品、調理状態が異なるときは入力値を修正します。具材の画像候補には使用量を設定しません。

## 実写真24枚での初回結果

長辺960px、CPU一スレッドで、原本のSHA-256を確認して実行しました。4枚だけを原本の目視で粗くトレースして可視マスクを採点しました。細い麺の輪郭は数ピクセルのずれでも数値が変わります。検出された候補面積がある画像は各方式とも24枚でした。

| 方式 | 平均IoU（4枚） | 背景混入率 | 見逃し率 | 2枚目以降の中央値 | プロセスのピークメモリー |
|---|---:|---:|---:|---:|---:|
| OpenCV HSV | 0.461 | 53.4% | 2.6% | 1.7 ms | 97 MiB |
| OpenCV HSV＋Lab | 0.711 | 27.2% | 3.8% | 74 ms | 104 MiB |
| MobileSAM | 0.589 | 40.6% | 0.6% | 3.08秒 | 761 MiB |

MobileSAMは見えている麺の欠けが少ない一方、分離した麺の間や反射を拾いました。この条件ではHSV＋Labの方がIoUが高く、MobileSAMを画像だけで重量を確定する標準方式にはしません。パスタが重なった場合は一枚の可視マスクから下の層の重量は分かりません。

| 訓練内の比較（22枚） | MAE | RMSE | 最大絶対誤差 |
|---|---:|---:|---:|
| MobileSAM 画像に占める面積率 | 33.68 g | 41.23 g | 90.53 g |
| MobileSAM 面積率＋形状 | 27.10 g | 38.46 g | 92.70 g |
| HSV＋Lab 画像に占める面積率 | 24.62 g | 28.92 g | 53.96 g |
| HSV＋Lab 面積率＋形状 | 15.92 g | 20.08 g | 49.59 g |
| 従来のOpenCV形状・質感など | 20.02 g | 25.45 g | 54.25 g |
| 背景色のみ | 28.85 g | 34.37 g | 69.51 g |

既存写真では有効なA4マーカーが0件でした。A4実寸でのA・B・Cの誤差は未測定です。SAM 2.1とDINOv3は接続試験に初期化重みを使用しました。SAM 3.1は一フレームの呼び出しと終了をモックで確認しました。この環境ではMeta公式の配布先から該当する事前学習重みを取得できず、CUDAも使えませんでした。独立実測試料でのMAE・RMSE・成功率は未測定です。

完成料理はOpenCLIPの公式重みで合成プレート3枚、乾麺の実写真1枚、空の入力1枚を動かしました。この5件はすべて候補表示を見送り、料理名を選べる状態になりました。3種類に対応したプロンプトと見える具材の候補を実装し、採択・見送り・修正をテストしました。正解を確認した完成料理の写真を用いた精度評価はまだです。
