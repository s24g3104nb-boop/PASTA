# 端末内画像認識と管理者アプリ

[基準シートなしの比較・追加撮影計画](44_SHEETLESS_WEIGHT.md)

通常アプリは、同梱のOpenCVで完成料理の写真からカルボナーラ・トマトパスタ・ジェノベーゼの候補を選びます。通常のセットアップで使えます。写真と標準レシピの候補を確認し、料理名・乾麺の重量・材料と分量を修正できます。

色解析ではHSVとLabの色条件を組み合わせ、写真中央に広がるソースの色を重視します。周辺にも同じ色が多い場合や、中央の一部にしか色がない場合は、その色の影響を抑えます。手がかりが弱くても3種類から1つを表示します。白皿・黒皿・木目背景、明暗、トッピングを組み合わせた合成画像で確認しています。ラベル付き実写真での比較精度は未確認です。

## iPhoneから記録する

1. Windowsで `setup_windows.bat` を実行し、通常は `start_windows.bat` で起動します。iPhoneからは同じ信頼できるWi-Fi上で `start_lan.bat` を起動し、SafariでPCのIPv4アドレスとポート8501を開きます。
2. ホームで「完成料理を記録」を開きます。家庭内のHTTP接続ではiPhoneの標準カメラで撮った写真をアップロードします。ブラウザ内カメラにはHTTPSかその端末のlocalhostとカメラ権限が必要です。
3. 新規の食事記録で写真を選ぶと、候補の標準レシピが自動で入ります。「標準レシピの料理」で候補を確認し、別の料理に変更できます。対象外の料理は標準レシピを使わず、料理名を入力して記録できます。
4. 保存した乾麺の測定を選ぶか、茹でる前の参考重量を入力します。初期値の80gは実測値やAI推定値ではありません。標準レシピの材料と量を確認し、必要に応じて編集・追加・削除します。
5. 料理全体と食べた分の栄養を確認し、食事記録を保存します。食事日記とカレンダーから見返せます。1件の乾麺記録は1件の食事に関連付けられます。

[家庭内LANの手順](../packaging/windows/SMARTPHONE_JA.md)

## 研究用OpenCLIPをWindowsで準備する

過去のOpenCLIP方式をローカルで評価・比較する手順です。Python 3.12の64ビットx64版を使います。PowerShellでPasta Lensを展開したフォルダーに移動し、`setup_windows.bat` で通常のアプリをセットアップします。研究を行う場合は、以下のCPU版の追加インストールをそのアプリの `.venv` に行います。

```powershell
.\setup_windows.bat
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1
.\.venv\Scripts\python.exe -m pip install -r pasta_weight_app_v2/requirements-local-vision.txt
.\.venv\Scripts\python.exe -m pip check
```

[公式のOpenCLIP重み](https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt)をPC内の非公開フォルダーへ保存します。使うモデルはViT-B/32 QuickGELU LAION400M E32で、重みは約605 MBです。下のパスを保存先に合わせ、SHA-256が一致したことを確認して同じPowerShellで設定します。写真の推論はPC内で行われます。

```powershell
$Model = 'C:\PastaLensModels\vit_b_32-quickgelu-laion400m_e32-46683a32.pt'
$Expected = '46683a32721d5c68911153698992361285d20ca690bb4f317c11e45c03d798fa'
if ((Get-FileHash -LiteralPath $Model -Algorithm SHA256 -ErrorAction Stop).Hash -ne $Expected) {
    throw 'OpenCLIPの重みが公式ファイルと一致しません。ダウンロード元と保存先を確認してください。'
} else {
    $env:PASTA_OPENCLIP_CHECKPOINT = $Model
    $env:PASTA_LOCAL_VISION_DEVICE = 'cpu'
}
```

### Windowsで研究用OpenCLIPの推論時間とメモリを測る

自分で撮ったJPEGまたはPNGの写真を指定します。`$Photo` を手元のパスに変更してください。診断は写真を変更せず、時刻やメモリの測定結果を新規の一時フォルダーに保存します。

```powershell
$Photo = 'C:\PastaPrivate\photo.jpg'
$CheckDir = Join-Path $env:TEMP ('pasta-dish-check-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $CheckDir | Out-Null
$Report = Join-Path $CheckDir 'dish-profile.json'
.\.venv\Scripts\python.exe -m pasta_weight_app_v2.experiments.bench_dish_vision --checkpoint "$env:PASTA_OPENCLIP_CHECKPOINT" --photo "$Photo" --output "$Report" --device cpu
if ($LASTEXITCODE -ne 0) { throw '診断に失敗しました。上のエラーを確認してください。' }
Get-Content -LiteralPath $Report -Raw
```

| JSONの項目 | 確認する値 |
|---|---|
| `platform` / `torch_version` / `device` | OS、PyTorchの版、実行デバイス |
| `cold_load_s` | 新しいプロセスでモデルを読み込む秒数。SHA-256の検証を含む |
| `decode_s` | 画像を開いて推論用に整える秒数 |
| `first_inference_s` / `warm_inference_median_s` | 初回の推論と、続く5回の推論時間の中央値 |
| `host_process_peak_rss_bytes` | 診断プロセスのピーク実メモリ。Windowsではピークワーキングセットで、MiBへは1,048,576で割る。取得不可なら `null` |
| `prediction` | 料理候補または判別不能 |

CLIの時間にはブラウザーとの転送を含みません。Windows実機での導入、推論時間、メモリ使用量は未確認です。

## Windows実機で完成料理の記録を確認する

本番の食事記録と分けて確認するには、新しい一時フォルダー内に検証用の保存先を指定します。既存アプリを終了してから、通常のセットアップが済んだフォルダーで起動します。

```powershell
$CheckDir = Join-Path $env:TEMP ('pasta-dish-check-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $CheckDir | Out-Null
$env:PASTA_DATA_DIR = Join-Path $CheckDir 'app-data'
$env:PASTA_RESEARCH_DATA_DIR = Join-Path $CheckDir 'research-data'
.\start_windows.bat
```

1. ホームの「完成料理を記録」で、自分で調理したカルボナーラ・トマトパスタ・ジェノベーゼのいずれかの写真を選びます。候補の標準レシピが入ることを確認します。
2. 「標準レシピの料理」で候補を確認し、別の料理に変更できることを確認します。編集済みのレシピを切り替える場合は、表示される適用ボタンを使います。
3. 「計算に使う乾麺」で参考重量を入力します。保存済み重量でも試す場合は、ホームで実測重量を新規に保存してからその記録を選びます。料理候補は材料や実際の使用量を決めないので、材料を編集して確認します。
4. 全体と食べた分の栄養が更新されたら「食事記録を保存」を押します。「記録」の食事日記とカレンダーを開き、料理名、栄養、写真が残ることを確認します。
5. 写真なしの新規記録でも、料理名の入力や標準レシピの手動選択から保存できることを確認します。

確認後に検証用PowerShellを閉じると、そのウィンドウで設定した検証用保存先は別の起動に引き継がれません。本番利用では、通常の保存先で `start_windows.bat` を起動します。

## 乾麺のMobileSAMを追加する場合

研究用の乾麺画像抽出にMobileSAMを使う場合は、別途固定したソースをインストールし、公式の重みを指定します。

```powershell
.\.venv\Scripts\python.exe -m pip install "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git@f706ad9c4eb7f219c00d9050e46328518ffb65d2"
$env:PASTA_MOBILESAM_CHECKPOINT = 'C:\PastaLensModels\mobile_sam.pt'
```

MobileSAM ViT-Tの重みは約41 MBです。CUDAを使う場合は、CUDA対応のPyTorchと対応するtorchvisionをインストールし、`PASTA_LOCAL_VISION_DEVICE=cuda` を設定します。iPhoneから接続する場合は公式重みを設定した同じPowerShellで `.\start_lan.bat` を起動します。 [MobileSAM公式重み](https://github.com/ChaoningZhang/MobileSAM/blob/f706ad9c4eb7f219c00d9050e46328518ffb65d2/weights/mobile_sam.pt) · [MobileSAMのライセンス](https://github.com/ChaoningZhang/MobileSAM/blob/master/LICENSE)

## 管理者の起動方法

管理者アプリには通常版と重ならない専用フォルダーを指定します。今ある研究用フォルダーを使う場合は、正常なバックアップから別の場所へコピーし、通常データや旧ファイルを上書きせずに指定します。通常版から管理者への画面遷移はありません。

```powershell
$env:PASTA_ADMIN_DATA_DIR = "C:\PastaLensAdmin\admin_data"
$secret = Read-Host "管理用パスワードを入力（12文字以上）" -AsSecureString
$env:PASTA_ADMIN_PASSWORD = [System.Net.NetworkCredential]::new('', $secret).Password
.\.venv\Scripts\python.exe -m streamlit run pasta_weight_app_v2/admin_app.py --server.address=127.0.0.1 --server.port=8502
```

管理者画面は自分のPCの `http://127.0.0.1:8502` でパスワードを入力して利用します。LAN待ち受けで起動した場合はログイン画面を出す前に拒否します。管理者プロセスを公開プロキシやトンネルの接続先に指定しないでください。通常と管理者のSQLite・写真・研究領域・ログは別です。必要な場合は `PASTA_ADMIN_RESEARCH_DIR` で管理者の研究領域を指定できます。

## 研究用OpenCLIPの学習・評価

OpenCLIPの画像エンコーダーは固定し、調理回ごとに集約した画像特徴から3クラスの線形分類層だけを学習します。初期の比較では既存のゼロショット方式と同じ検証・テスト写真を使います。研究評価では、学習済み分類層を設定しても、分類スコアの差が小さい場合やゼロショット側で対象外・判別困難と判定した場合は判別を見送ります。

1. 元写真と元ラベルは非公開フォルダーに保存します。確認済みの料理名、調理回、写真のSHA-256から `file,label,session_id,sha256,split` のUTF-8 CSVを作ります。`file` は写真フォルダーからの相対パス、`label` は「カルボナーラ」「トマトパスタ」「ジェノベーゼ」のいずれかです。
2. 同じ皿の別角度や連写には同じ `session_id` を付けます。同じ調理回の写真は `train`・`validation`・`test` のいずれか一つにまとめます。各分割に3種類がそろうよう、料理ごとに別の調理回を確保してください。調理回が不明な写真は確認が取れるまで台帳に入れません。同じ料理の撮り直しや拡大画像を別の調理回として扱わないでください。
3. 対象外への誤分類も調べる場合は、別の調理回で撮影した対象外料理や乾麺などを `validation`・`test` に `対象外` として登録します。対象外の写真がなければ、その見送り率は未測定として出力されます。
4. 次のコマンドで学習と評価を行います。`metrics.json` に写真数と調理回数、クラス別正解率・適合率・再現率・F1、判別不能列を含む混同行列、候補を表示した割合、調理回ごとの正解率の平均を分割別に出します。未表示の写真も分母に含めます。テスト結果を見て方式やしきい値を選び直した場合は、新しい調理回で確認します。

```powershell
.\.venv\Scripts\python.exe -m pasta_weight_app_v2.experiments.train_finished_dish `
  --manifest C:\PastaPrivate\dishes.csv --photos C:\PastaPrivate\photos `
  --checkpoint $env:PASTA_OPENCLIP_CHECKPOINT --output C:\PastaPrivate\models\dish-run-01
```

公式OpenCLIPのゼロショット推論は、このCLI比較で基準に使います。写真、台帳、追加学習の重みと評価結果は非公開フォルダーに保管してください。

### 受領台帳の照合と追加学習の比較

従来のCSVに加え、分割ZIPの `labels.csv` と `split_manifest.csv` を同じ非公開フォルダーに展開した場合は、以下の手順で照合できます。変換済み画像のハッシュ・サイズ・デコード結果を確認し、未変換画像は元画像ハッシュにも照合します。HEIC原本は変換済みJPEGから照合できないため、未照合枚数を別に記録します。

目視レビューCSVは `file,group,split,group_status` を含めます。`file` は台帳にある `images/…` のパスです。同一の皿・連写・転載や切り抜きの疑いがある組は同じ群に入れます。目視や画像類似度で推定した群は `inferred` を指定します。分割内に3クラスが必要で、明らかな類似群が残る場合は分割を修正してから学習を始めます。画像から群を推定した結果は探索的評価として扱います。照合で生成する分割CSVにも確度を引き継ぎます。

```powershell
.\.venv\Scripts\python.exe -m pasta_weight_app_v2.experiments.audit_finished_dish `
  --dataset C:\PastaPrivate\bundle --review C:\PastaPrivate\visual_review.csv `
  --output C:\PastaPrivate\verified
.\.venv\Scripts\python.exe -m pasta_weight_app_v2.experiments.train_finished_dish `
  --manifest C:\PastaPrivate\verified\split.csv --photos C:\PastaPrivate\bundle `
  --checkpoint $env:PASTA_OPENCLIP_CHECKPOINT --output C:\PastaPrivate\comparison-01 `
  --group-status inferred --fine-tune
```

`--fine-tune` は画像エンコーダーの最終Transformerブロックと3クラスの分類層を、学習用の写真だけで固定5エポック調整します。同じ群やクラスの写真数だけで学習が偏らないよう、クラスと群ごとの寄与を揃えます。学習後は同じデータ分割で、従来のゼロショット、固定特徴＋分類層、最終ブロック調整の3方式を比較します。学習方式の候補表示には既存のゼロショット見送り判定も適用します。

`metrics.json` の `protocol.group_status` は評価の来歴、`predictions.csv` は写真ごとの予測です。変換前原本、写真、台帳、予測、調整した重みは非公開で保管します。アプリ本体は `fine_tuned_block.pt` を自動で読み込みません。台帳の正解率が高い場合も、調理回・転載元を確認した別の写真と対象外料理で評価するまで、自動判別の適用範囲は確認できません。

### 撮影条件と対象外画像の評価

`bench_dish_vision --dataset PRIVATE_DATASET --checkpoint OFFICIAL_WEIGHTS --output PRIVATE_RESULT.json` は、検証済み台帳の全写真に固定した露出、色かぶり、回転、射影変形、縮小、明暗の余白を適用し、正解、誤った候補、判別不能、原画像からの変化を記録します。余白は画像全体に同じ幅の枠を加える試験で、背景を実際に交換する試験には別撮影が必要です。回転・射影変形で別角度の遮蔽や反射を再現できる範囲は限られます。

利用できる対象外画像がある場合は、`--controls PRIVATE_CONTROLS.csv --control-root PRIVATE_ROOT` を追加します。列は `file,kind,sha256` です。乾麺やその他の物体と、対象外の完成料理には別の `kind` を付け、誤候補率をそれぞれ集計します。候補表示のしきい値は同じままで比較します。料理名の候補は、重量・材料・実際の使用量の測定には使いません。

撮影条件の変換を省き、両方の割合を同じ比較で確認する場合は `--original-only` を追加します。

- `by_condition.original.false_reject_rate` は、正解が3種類に含まれる全写真に対する判別不能の割合です。別の候補を出した件数は `wrong_candidate` を確認します。`by_class` には料理別の結果があります。
- `controls` の各 `false_accept_rate` は、その対象外グループの全写真に対し、3種類のいずれかを誤って候補に出した割合です。`rejection_rate` は同じグループで判別を見送った割合です。対象外写真を指定しなかった場合は、`all_controls` の割合が `null` になります。

対象外の完成料理には、評価に利用できる権利を確認した写真を使い、料理名と出典・利用条件を台帳に残します。ペペロンチーノ・クリームパスタ・ナポリタンなどの結果は料理別に確認してください。乾麺だけの見送り率では、別の完成料理の誤候補率は分かりません。

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

完成料理の初回接続確認では、OpenCLIPの公式重みで合成プレート3枚、乾麺の実写真1枚、空の入力1枚を動かしました。この5件はすべて候補表示を見送り、手動で料理名を選べました。その後の実写真評価の実行手順は「撮影条件と対象外画像の評価」を参照してください。
