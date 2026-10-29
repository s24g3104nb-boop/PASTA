/* Isolated CG-only Chromium acceptance test. No private database/photos are read. */
const {chromium}=require('playwright');
const {spawn,execFileSync}=require('child_process');
const fs=require('fs'),path=require('path'),os=require('os');
const root=path.resolve(__dirname,'..');
const work=process.env.PASTA_UI_ARTIFACTS||fs.mkdtempSync(path.join(os.tmpdir(),'pasta-diary-ui-'));
fs.mkdirSync(work,{recursive:true});
const python=process.env.PASTA_PYTHON||path.join(root,'.venv/bin/python');
const db=path.join(work,'data/measurements.sqlite3');
const port=process.env.PASTA_UI_PORT||'8523';
const proc=spawn(python,['-m','streamlit','run','pasta_weight_app_v2/app.py','--server.address','127.0.0.1','--server.port',port,'--server.headless','true','--browser.gatherUsageStats','false'],{cwd:root,env:{...process.env,PASTA_DATA_DIR:path.join(work,'data')},stdio:['ignore',fs.openSync(path.join(work,'server.log'),'a'),fs.openSync(path.join(work,'server.log'),'a')]});
let browser;const results=[];
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function ready(p){await pause(450);await p.waitForFunction(()=>!document.querySelector('[data-stale="true"]'));if(await p.locator('[data-testid=stException]').count())throw Error(await p.locator('[data-testid=stException]').innerText());}
async function inspect(p,label){await ready(p);const d=await p.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,main:document.querySelector('[data-testid=stMain]')?.scrollWidth}));if(d.scroll>d.width+1||d.main>d.width+1)throw Error(label+' overflow '+JSON.stringify(d));results.push(label);console.log('PASS '+label);await p.screenshot({path:path.join(work,label+'.png'),fullPage:true});}
async function click(p,name){await p.getByRole('button',{name,exact:true}).click();await ready(p);}
async function nav(p,name){await p.locator('.st-key-main_nav [data-testid=stRadioOption]').filter({hasText:name}).click();await ready(p);}
async function select(p,name,value){await p.getByRole('combobox',{name,exact:true}).click();await p.getByRole('option',{name:value,exact:true}).click();await ready(p);}
function sql(statement){execFileSync(python,['-c','import sqlite3,sys\nwith sqlite3.connect(sys.argv[1]) as c:c.executescript(sys.argv[2])',db,statement]);}
(async()=>{
for(let i=0;i<100;i++){try{if((await fetch('http://127.0.0.1:'+port+'/_stcore/health')).ok)break;}catch{}await pause(200);}
browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
const p=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true,colorScheme:'light'});
await p.goto('http://127.0.0.1:'+port);await p.getByRole('button',{name:'乾麺の重さをはかる',exact:true}).waitFor();await inspect(p,'01-home-light');
await click(p,'乾麺の重さをはかる');await inspect(p,'02-camera');
await click(p,'カメラが開かない場合は写真から選ぶ');
await p.locator('input[type=file]').setInputFiles({name:'broken.png',mimeType:'image/png',buffer:Buffer.from('broken')});await ready(p);await p.getByText('写真を読み込めませんでした。',{exact:false}).waitFor();await inspect(p,'03-upload-error');
await click(p,'撮り直す・写真を選び直す');await p.locator('input[type=file]').setInputFiles(path.join(root,'pasta_weight_app_v2/sample_images/demo_spread_12.png'));await ready(p);await inspect(p,'04-upload-preview');
await click(p,'この写真で解析する');await p.getByText('実製品の1本重量が未校正です。',{exact:false}).waitFor();await inspect(p,'05-calibration-rejection');
await nav(p,'設定');await select(p,'設定メニュー','開発・検証ツール');await click(p,'サンプルで試す');await inspect(p,'06-cg-preview');
await click(p,'この写真で解析する');await inspect(p,'07-weight-result');
await p.getByText('実測重量も記録する（任意）',{exact:true}).click();await p.getByRole('spinbutton',{name:'実測重量（g）',exact:true}).fill('40');await p.getByRole('spinbutton',{name:'実測重量（g）',exact:true}).press('Tab');await ready(p);
await click(p,'履歴に保存');await p.getByRole('button',{name:'この乾麺で料理を記録する',exact:true}).waitFor();if(!await p.getByRole('button',{name:'履歴に保存',exact:true}).isDisabled())throw Error('save not disabled');
await click(p,'この乾麺で料理を記録する');await p.getByRole('textbox',{name:'料理名',exact:true}).fill('テストのトマトパスタ');await p.getByRole('textbox',{name:'ひとこと（任意）',exact:true}).fill('CGによる授業用テスト');await p.getByRole('textbox',{name:'ひとこと（任意）',exact:true}).press('Tab');await ready(p);
await p.locator('input[type=file]').setInputFiles(path.join(root,'pasta_weight_app_v2/sample_images/demo_spread_12.png'));await ready(p);await inspect(p,'08-meal-editor-photo');
await nav(p,'ホーム');await click(p,'食べたパスタを記録する');if(await p.getByRole('textbox',{name:'料理名',exact:true}).inputValue()!=='テストのトマトパスタ')throw Error('draft lost');await inspect(p,'09-draft-restored');
sql("CREATE TRIGGER reject_test BEFORE INSERT ON meal_entries BEGIN SELECT RAISE(ABORT,'simulated disk failure'); END;");
await click(p,'食事記録を保存');await p.getByText('保存できませんでした。入力内容は残っています。',{exact:false}).waitFor();await inspect(p,'10-save-failure-retained');
sql('DROP TRIGGER reject_test;');await click(p,'食事記録を保存');await p.getByRole('button',{name:'記録を編集する',exact:true}).waitFor();await inspect(p,'11-meal-detail');
await click(p,'記録を編集する');await p.getByRole('textbox',{name:'料理名',exact:true}).fill('編集後のパスタ');await p.getByRole('textbox',{name:'料理名',exact:true}).press('Tab');await click(p,'食事記録を保存');await p.getByRole('heading',{name:'編集後のパスタ',exact:true}).waitFor();await inspect(p,'12-edited-detail');
await click(p,'栄養計算を追加・変更する');
await p.getByText('この記録の乾麺を今回の料理に使いました',{exact:true}).click();await ready(p);
await p.getByText('使った追加材料・ソース・油・調味料をすべて入力しました（未入力分は合計に含まれません）',{exact:true}).click();await ready(p);
await click(p,'入力を確認して計算・保存');await inspect(p,'12b-cooking-calculation');
await p.getByRole('button',{name:'この料理を食事日記に記録する',exact:true}).first().click();await ready(p);await click(p,'食事記録を保存');await inspect(p,'12c-linked-nutrition');
await click(p,'記録一覧へ戻る');await inspect(p,'13-photo-list');await p.getByRole('textbox',{name:'料理名で検索',exact:true}).fill('見つからない');await p.getByRole('textbox',{name:'料理名で検索',exact:true}).press('Tab');await ready(p);await p.getByText('0件の食事記録',{exact:true}).waitFor();
await p.getByRole('textbox',{name:'料理名で検索',exact:true}).fill('');await p.getByRole('textbox',{name:'料理名で検索',exact:true}).press('Tab');await ready(p);
await p.getByText('カレンダー',{exact:true}).click();await ready(p);await inspect(p,'14-calendar');
await click(p,'この日に食事を記録する');await p.getByRole('textbox',{name:'料理名',exact:true}).fill('写真なしのパスタ');await p.getByRole('textbox',{name:'料理名',exact:true}).press('Tab');await click(p,'食事記録を保存');await p.getByText('栄養：未計算',{exact:true}).waitFor();await inspect(p,'15-photo-free-unknown');
await click(p,'記録一覧へ戻る');await inspect(p,'16-same-day-calendar');
await p.getByRole('button',{name:'この食事を見る',exact:true}).first().click();await ready(p);await p.getByText('この記録を削除する',{exact:true}).click();await p.getByText('この食事記録を一覧から削除する',{exact:true}).click();await click(p,'削除を確定');await p.getByText('食事記録を削除しました。',{exact:true}).waitFor();await inspect(p,'17-after-delete');
for(const width of [375,430,768,1440]){await p.setViewportSize({width,height:900});await inspect(p,'18-calendar-'+width);}
await nav(p,'設定');await select(p,'設定メニュー','表示・使い方');await select(p,'カラーテーマ','ダーク');await inspect(p,'19-settings-dark');await nav(p,'ホーム');await inspect(p,'20-home-dark');await nav(p,'記録');await inspect(p,'21-records-dark');
console.log(JSON.stringify({passed:results.length,results,artifacts:work},null,2));fs.writeFileSync(path.join(work,'results.json'),JSON.stringify({passed:results.length,results},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();proc.kill('SIGTERM');});
