/* Chromium integration check with a disposable database and synthetic fixtures only.
 * Run: PASTA_TEST_PYTHON=.venv/bin/python node tools/browser_integration.cjs [artifact-directory]
 * Requires Playwright and Chromium; never opens the user's data directory.
 */
const {chromium}=require('playwright');
const fs=require('fs'),os=require('os'),path=require('path');
const {spawn,execFileSync}=require('child_process');
const root=path.resolve(__dirname,'..');
const python=path.resolve(root,process.env.PASTA_TEST_PYTHON||'.venv/bin/python');
const work=fs.mkdtempSync(path.join(os.tmpdir(),'pasta-integration-'));
const out=path.resolve(process.argv[2]||path.join(work,'screenshots'));
fs.mkdirSync(out,{recursive:true});
const env={...process.env,PASTA_DATA_DIR:path.join(work,'data'),PASTA_RESEARCH_DATA_DIR:path.join(work,'research')};
const port=Number(process.env.PASTA_TEST_PORT||8772), url=`http://127.0.0.1:${port}`;
const checks=[];
function py(code){return execFileSync(python,['-c',code],{cwd:root,env,encoding:'utf8'}).trim()}
py(`import sys,os\nfrom pathlib import Path\nsys.path.insert(0,'pasta_weight_app_v2')\nfrom storage import init_db,save_product\nfrom experiments.capture_sheet import render_sheet\nfrom PIL import Image\ninit_db()\nfor name,g,n in [('検証用・未校正（合成テスト）',0,(350,12,2,70)),('検証用・校正済み（合成テスト）',.68,(350,12,2,70)),('検証用・栄養未登録（合成テスト）',0,(0,0,0,0))]:save_product(name,g,*n)\np=Path(os.environ['PASTA_DATA_DIR']).parent\na=render_sheet(2)\na[130:460,150:260]=(220,180,40)\nImage.fromarray(a).save(p/'sheet.png')\nImage.new('RGB',(800,600),'white').save(p/'blank.png')\n(p/'broken.png').write_bytes(b'not an image')`);
// Refuse an occupied port, so this harness can never attach to a running user app.
execFileSync(python,['-c',`import socket\ns=socket.socket()\ns.bind(('127.0.0.1',${port}))\ns.close()`],{cwd:root,env});
const serverLog=fs.openSync(path.join(out,'server.log'),'w');
const server=spawn(python,['-m','streamlit','run','pasta_weight_app_v2/app.py','--server.port',String(port),'--server.address','127.0.0.1','--server.headless','true','--browser.gatherUsageStats','false'],{cwd:root,env,stdio:['ignore',serverLog,serverLog]});
const image=path.join(root,'pasta_weight_app_v2/sample_images/demo_spread_12.png');
const wait=ms=>new Promise(r=>setTimeout(r,ms));
async function settle(p){await p.waitForTimeout(650)}
async function click(p,key){await p.locator('.st-key-'+key).getByRole('button').click();
  await settle(p)}
async function nav(p,name){await p.locator('.st-key-main_nav label').filter({hasText:name}).click();
  await settle(p)}
async function selectIndex(p,key,index){const cb=p.locator('.st-key-'+key).getByRole('combobox');
  await cb.focus();
  await cb.press('ArrowDown');
  await cb.press('Home');for(let i=0;i<index;i++)await cb.press('ArrowDown');
  await cb.press('Enter');
  await settle(p)}
async function input(p,role,name,value){const e=p.getByRole(role,{name,exact:true});
  await e.fill(value);
  await e.press('Tab');
  await settle(p)}
async function shot(p,name){await settle(p);
  await p.screenshot({path:path.join(out,name+'.png'),fullPage:true});
  if(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2))throw Error('horizontal overflow '+name);
  if(await p.getByTestId('stException').count())throw Error('unhandled Streamlit exception '+name)}
async function fixedCaptureTime(p){const t=p.getByTestId('stTimeInput');
  await t.locator('[data-type=hour]').press('1');
  await t.locator('[data-type=hour]').press('2');
  await t.locator('[data-type=minute]').press('0');
  await t.locator('[data-type=minute]').press('0');
  await p.getByRole('heading',{name:'独立試料・撮影データ登録',exact:true}).click();
  await settle(p)}
async function a4Reference(p){const cb=p.getByTestId('stSelectbox').filter({hasText:'写真内のスケール基準・撮影条件'}).getByRole('combobox');
  await cb.focus();
  await cb.press('ArrowDown');
  await cb.press('Home');
  await cb.press('ArrowDown');
  await cb.press('Enter');
  await settle(p)}
async function open(p){await p.goto(url);
  await p.locator('.st-key-home_camera button').waitFor();
  await settle(p)}
async function upload(p,file=image){await click(p,'home_camera');
  await click(p,'camera_fallback');
  await p.locator('input[type=file]').setInputFiles(file);
  await settle(p)}
async function research(p){await nav(p,'設定');
  await selectIndex(p,'settings_section',3);
  await selectIndex(p,'developer_section',2);
  await p.getByRole('heading',{name:'独立試料・撮影データ登録',exact:true}).waitFor()}
function sql(db,stmt){return py(`import sqlite3\nc=sqlite3.connect(${JSON.stringify(path.join(work,db))})\nc.execute(${JSON.stringify(stmt)})\nc.commit()\nc.close()`)}
async function caseRun(b,name,width,mode,fn){const p=await b.newPage({viewport:{width,height:900},colorScheme:mode});p.setDefaultTimeout(18000);try{await open(p);
  await fn(p);checks.push({case:name,passed:true})}catch(e){checks.push({case:name,passed:false,error:e.message});
  await p.screenshot({path:path.join(out,'failure_'+name.replace(/[^a-z0-9]/gi,'_')+'.png'),fullPage:true});fs.writeFileSync(path.join(out,'failure_'+name.replace(/[^a-z0-9]/gi,'_')+'.txt'),await p.locator('body').innerText())}finally{await p.close()}console.log(JSON.stringify(checks.at(-1)))}
(async()=>{let b;try{
for(let n=0;n<60;n++){try{if((await fetch(url+'/_stcore/health')).ok)break}catch{}if(n===59)throw Error('server startup timeout');
  await wait(500)}
b=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',args:['--no-sandbox','--disable-dev-shm-usage']});
for(const width of [1280,390])for(const mode of ['light','dark'])await caseRun(b,`display_${width}_${mode}`,width,mode,async p=>{
await shot(p,`home_${width}_${mode}`);
  await upload(p);
  await selectIndex(p,'which_product',2);
  await shot(p,`capture_${width}_${mode}`);
  await click(p,'analyze_photo');
  await p.getByText('推定重量',{exact:true}).waitFor();
  await shot(p,`result_${width}_${mode}`);
  await nav(p,'記録');
  await shot(p,`diary_${width}_${mode}`);
  await nav(p,'設定');
  await shot(p,`settings_${width}_${mode}`);
  await research(p);
  await shot(p,`research_${width}_${mode}`);
await p.getByText('撮影の準備・A4シートの使い方',{exact:true}).click();
  const dl=p.waitForEvent('download');
  await p.getByRole('button',{name:'A4撮影補助シートをダウンロード',exact:true}).click();await(await dl).saveAs(path.join(out,`sheet_${width}_${mode}.pdf`));
});
await caseRun(b,'manual100_cooking_diary_edit_delete',390,'light',async p=>{
await upload(p);
  await selectIndex(p,'which_product',1);
  await click(p,'analyze_photo');
  await p.locator('.st-key-manual_weight_save button').waitFor();
  await shot(p,'uncalibrated_recovery');
  await input(p,'spinbutton','はかりで測った実測重量（g）','100');
  await p.getByText('350.0 kcal',{exact:true}).waitFor();
  await shot(p,'manual100_nutrition');
sql('data/measurements.sqlite3',"CREATE TRIGGER browser_fail BEFORE INSERT ON measurements BEGIN SELECT RAISE(ABORT,'synthetic save failure'); END");
  await click(p,'manual_weight_save');
  await p.getByText(/保存できませんでした。写真と入力値は残っています/).waitFor();
  await shot(p,'manual_save_failure');sql('data/measurements.sqlite3','DROP TRIGGER browser_fail');
  await click(p,'manual_weight_save');
  await p.getByText('実測重量を保存しました。推定重量は記録していません。',{exact:true}).waitFor();
  if(!await p.locator('.st-key-manual_weight_save button').isDisabled())throw Error('duplicate submit not disabled');
  await click(p,'manual_weight_to_diary');
await input(p,'textbox','料理名','合成テスト・100gの料理');
  await p.getByRole('button',{name:'材料を入力して栄養を計算する',exact:true}).click();
  await settle(p);
  await p.getByText('使用する乾麺: 100.00g / 実測重量',{exact:true}).waitFor();
  await p.getByText('この記録の乾麺を今回の料理に使いました',{exact:true}).click();
  await settle(p);
  await p.getByText('使った追加材料・ソース・油・調味料をすべて入力しました（未入力分は合計に含まれません）',{exact:true}).click();
  await settle(p);
  await p.getByRole('button',{name:'入力を確認して計算・保存',exact:true}).click();
  await settle(p);
  await p.locator('.st-key-cooking_saved_to_diary button').waitFor();
  await shot(p,'cooking100_result');
  await click(p,'cooking_saved_to_diary');
  await p.getByRole('textbox',{name:'料理名',exact:true}).waitFor();
  if(await p.getByRole('textbox',{name:'料理名',exact:true}).inputValue()!=='合成テスト・100gの料理')throw Error('diary draft lost in cooking');
  await p.locator('input[type=file]').setInputFiles(image);
  await settle(p);
  await click(p,'meal_save');
  await p.locator('.st-key-meal_edit button').waitFor();
  await shot(p,'diary_detail_photo');
  await click(p,'meal_edit');
  await input(p,'textbox','料理名','編集済み・合成100g');
  await click(p,'meal_save');
  await p.getByText('編集済み・合成100g',{exact:true}).waitFor();
  await click(p,'meal_detail_back');
  await p.getByText('カレンダー',{exact:true}).click();
  await settle(p);
  await shot(p,'diary_calendar');
  await p.getByText('写真付き一覧',{exact:true}).click();
  await input(p,'textbox','料理名で検索','編集済み');
  await shot(p,'diary_search');
  await p.getByRole('button',{name:'この食事を見る',exact:true}).click();
  await settle(p);
  await p.getByText('この記録を削除する',{exact:true}).click();
  await p.getByText('この食事記録を一覧から削除する',{exact:true}).click();
  await settle(p);
  await click(p,'meal_delete');
  await p.getByText('食事記録を削除しました。',{exact:true}).waitFor();
});
await caseRun(b,'photo_free_unknown_nutrition_diary',390,'dark',async p=>{
await p.getByText('写真を選ぶ・はかりの数値を記録する',{exact:true}).click();
  await click(p,'home_manual_weight');
  await selectIndex(p,'manual_only_product',3);
  await input(p,'spinbutton','はかりで測った実測重量（g）','80');
  await p.getByText(/栄養は未計算です/).waitFor();
  await click(p,'manual_weight_save');
  await click(p,'manual_weight_to_diary');
  await input(p,'textbox','料理名','写真なし・未計算の合成記録');
  await click(p,'meal_save');
  await p.getByText('栄養：未計算',{exact:true}).waitFor();
  await shot(p,'photo_free_unknown_diary');
});
await caseRun(b,'unreadable_no_detection_missing_model',1280,'light',async p=>{
await upload(p,path.join(work,'broken.png'));
  await p.getByText(/写真を読み込めませんでした/).waitFor();
  await shot(p,'unreadable_recovery');
  await click(p,'decode_retake');
  await p.locator('input[type=file]').setInputFiles(path.join(work,'blank.png'));
  await settle(p);
  await selectIndex(p,'which_product',2);
  await click(p,'analyze_photo');
  await p.locator('.st-key-manual_weight_save button').waitFor();
  await shot(p,'blank_photo_recovery');
  await nav(p,'設定');
  await selectIndex(p,'settings_section',3);
  await p.getByText('測定で開発用設定を使う',{exact:true}).click();
  await settle(p);
  await p.getByText('学習モデル（実験用回帰）',{exact:true}).click();
  await settle(p);
  await click(p,'home_demo');
  await click(p,'analyze_photo');
  await p.getByText(/有効なモデルがありません/).waitFor();
  await p.locator('.st-key-manual_weight_save button').waitFor();
  await shot(p,'missing_model_recovery');
});
await caseRun(b,'research_qc_draft_failure_retry_duplicate',1280,'dark',async p=>{
await research(p);
  await input(p,'spinbutton','はかりに表示された実測重量（g）','34');
  await input(p,'textbox','パスタの製品名','合成テスト・研究用');
  await p.locator('input[type=file]').setInputFiles(path.join(work,'sheet.png'));
  await settle(p);
  await p.getByText('写真の確認結果',{exact:true}).waitFor();
  await a4Reference(p);
  await fixedCaptureTime(p);
  await shot(p,'research_markers');
  await nav(p,'ホーム');
  await research(p);
  if(await p.getByRole('spinbutton',{name:'はかりに表示された実測重量（g）'}).inputValue()!=='34.00')throw Error('research draft lost');
  await p.getByText('保存予定の写真',{exact:true}).waitFor();sql('research/collection.sqlite3',"CREATE TRIGGER browser_fail BEFORE INSERT ON entries BEGIN SELECT RAISE(ABORT,'synthetic failure'); END");
  await click(p,'collection_save');
  await p.getByText(/保存できませんでした。入力は保持しています/).waitFor();sql('research/collection.sqlite3','DROP TRIGGER browser_fail');
  await click(p,'collection_save');
  await p.getByText('写真と実測ラベルを保存しました。',{exact:true}).waitFor();
  await shot(p,'research_saved');
  await click(p,'collection_next_photo');
  await p.locator('input[type=file]').setInputFiles(path.join(work,'sheet.png'));
  await settle(p);
  await click(p,'collection_save');
  await p.getByText('登録済み写真または送信IDと内容が異なります。元データは変更せず、差分を監査履歴に保存しました。',{exact:true}).waitFor();
  await shot(p,'research_conflict');
  await a4Reference(p);
  await fixedCaptureTime(p);
  await click(p,'collection_save');
  await p.getByText('登録済みの写真です。重複登録は行いませんでした。',{exact:true}).waitFor();
  await shot(p,'research_duplicate');
});
}catch(e){checks.push({case:'infrastructure',passed:false,error:e.message})}finally{if(b)await b.close();server.kill('SIGINT');fs.closeSync(serverLog);
  const report={checks,all_passed:checks.every(x=>x.passed),fixture:'synthetic only; no real weight accuracy evaluation',real_windows:false,real_iphone:false,work,artifacts:out};fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2));
  const files=fs.readdirSync(out).filter(x=>x.endsWith('.png')&&!x.startsWith('failure_'));fs.writeFileSync(path.join(out,'index.html'),'<!doctype html><html lang="ja"><meta charset="utf-8"><title>Pasta Lens 統合検証</title><style>body{background:#faf8f2;color:#29352e;font:16px sans-serif;max-width:1100px;margin:auto}img{max-width:100%;border:1px solid #ddd}figure{background:white;padding:1rem}figcaption{margin:.5rem}</style><h1>Pasta Lens 統合検証</h1><p>合成画像・専用DBによるChromium検証。Windows/iPhone実機と重量精度は未検証。</p><pre>'+JSON.stringify(checks,null,2)+'</pre>'+files.map(f=>`<figure><figcaption>${f}</figcaption><a href="${f}"><img loading="lazy" src="${f}"></a></figure>`).join(''));console.log(JSON.stringify(report));
  if(!report.all_passed)process.exitCode=1}})();
