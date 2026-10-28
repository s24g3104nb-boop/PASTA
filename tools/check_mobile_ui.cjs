/* Optional browser regression: npm install playwright; CHROMIUM_PATH selects a local browser.
   Runs only with an isolated temporary dataset. Screenshots never enter the public repository. */
const {chromium}=require('playwright');
const {spawn}=require('child_process');
const fs=require('fs'),path=require('path'),os=require('os');
const root=path.resolve(__dirname,'..');
const work=fs.mkdtempSync(path.join(os.tmpdir(),'pasta-ui-'));
const python=process.env.PASTA_PYTHON||path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const port=process.env.PASTA_UI_PORT||'8523';
const proc=spawn(python,['-m','streamlit','run',path.join(root,'pasta_weight_app_v2/app.py'),'--server.address','127.0.0.1','--server.port',port,'--server.headless','true','--browser.gatherUsageStats','false'],{cwd:root,env:{...process.env,PASTA_DATA_DIR:path.join(work,'data')},stdio:'ignore'});
const pause=ms=>new Promise(r=>setTimeout(r,ms));
let browser;const results=[];
async function ready(page){await page.waitForFunction(()=>!document.querySelector('[data-testid="stException"]') && !document.querySelector('[data-stale="true"]'));await pause(250);}
async function nav(page,name){await page.locator('.st-key-main_nav [data-testid="stRadioOption"]').filter({hasText:name}).click();await ready(page);}
async function inspect(page,label){await ready(page);const dimensions=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,main:document.querySelector('[data-testid="stMain"]')?.scrollWidth}));if(dimensions.scroll>dimensions.width+1||dimensions.main>dimensions.width+1)throw Error(label+' horizontal overflow '+JSON.stringify(dimensions));if(await page.locator('[data-testid="stException"]').count())throw Error(label+' Streamlit exception');results.push({label,...dimensions});console.log("PASS "+label);}
(async()=>{
 for(let i=0;i<80;i++){try{if((await fetch(`http://127.0.0.1:${port}/_stcore/health`)).ok)break;}catch{}await pause(200);}
 browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||undefined,headless:true,args:['--no-sandbox']});
 for(const width of [375,390,430,768,1440]){
  const page=await browser.newPage({viewport:{width,height:900},isMobile:width<500,hasTouch:width<500,colorScheme:'light',reducedMotion:'reduce'});
  await page.goto(`http://127.0.0.1:${port}`);await page.getByRole('button',{name:'パスタを撮影する',exact:true}).waitFor();
  await inspect(page,`home-light-${width}`);await page.screenshot({path:path.join(work,`home-light-${width}.png`)});
  await nav(page,'設定');await page.getByRole('combobox',{name:'カラーテーマ',exact:true}).click();await page.getByRole('option',{name:'ダーク',exact:true}).click();await ready(page);await inspect(page,`settings-dark-${width}`);
  await nav(page,'ホーム');await page.getByRole('button',{name:'パスタを撮影する',exact:true}).waitFor();await inspect(page,`home-dark-${width}`);await page.screenshot({path:path.join(work,`home-dark-${width}.png`)});
  await nav(page,'履歴');await page.getByRole('heading',{name:'あなたの記録',exact:true}).waitFor();await inspect(page,`history-${width}`);
  await nav(page,'料理・栄養');await page.getByRole('heading',{name:'料理と栄養',exact:true}).waitFor();await inspect(page,`cooking-${width}`);
  await page.close();
 }
 const page=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});
 await page.goto(`http://127.0.0.1:${port}`);await page.getByText('写真なしで操作を試す',{exact:true}).click();await page.getByRole('button',{name:'サンプルで試す',exact:true}).click();
 await inspect(page,'preview');if(await page.getByRole('button',{name:'履歴に保存',exact:true}).count())throw Error('result before confirmation');
 await page.getByRole('button',{name:'この写真で解析する',exact:true}).click();await page.getByRole('button',{name:'履歴に保存',exact:true}).waitFor();await inspect(page,'result');
 await page.locator('.st-key-weight_hero').scrollIntoViewIfNeeded();await ready(page);await page.screenshot({path:path.join(work,'result.png')});
 await page.getByRole('button',{name:'履歴に保存',exact:true}).click();await page.getByText('測定履歴に保存しました。',{exact:false}).waitFor();
 await nav(page,'履歴');await page.getByText('詳細・重量の修正',{exact:true}).click();await page.getByRole('spinbutton',{name:'はかりで測った重量（g）',exact:true}).fill('46');
 await page.getByRole('button',{name:'実測重量を保存',exact:true}).click();await page.getByText('実測重量を保存しました。',{exact:true}).waitFor();await inspect(page,'history-edited');
 await nav(page,'ホーム');await page.getByRole('button',{name:'もう一度撮影',exact:true}).click();await page.getByRole('button',{name:'カメラが開かない場合は写真から選ぶ',exact:true}).waitFor();await inspect(page,'camera-denied-fallback');
 await page.getByRole('button',{name:'カメラが開かない場合は写真から選ぶ',exact:true}).click();
 await page.locator('input[type=file]').setInputFiles({name:'broken.png',mimeType:'image/png',buffer:Buffer.from('not-an-image')});await page.getByText('写真を読み込めませんでした。',{exact:false}).waitFor();await inspect(page,'invalid-upload');
 await page.getByRole('button',{name:'撮り直す・写真を選び直す',exact:true}).click();await page.getByText('写真は解析のためにこのアプリのサーバーへ送信されます。',{exact:false}).waitFor();await ready(page);await page.locator('input[type=file]').setInputFiles(path.join(root,'pasta_weight_app_v2/sample_images/demo_spread_12.png'));
 await page.getByRole('button',{name:'この写真で解析する',exact:true}).click();await page.getByText('実製品の1本重量が未校正です。',{exact:false}).waitFor();await inspect(page,'uploaded-photo-calibration-gate');
 // Native radio focus must still work after styling.
 await page.locator('.st-key-main_nav input[type=radio]').first().focus();await page.keyboard.press('ArrowRight');await page.getByRole('heading',{name:'あなたの記録',exact:true}).waitFor();
 await nav(page,'料理・栄養');await page.getByRole('combobox',{name:'この料理に使用した乾麺の計測履歴ID',exact:true}).click();await page.getByRole('option').filter({hasText:'ID 1 |'}).click();await page.getByText('食材を1つ追加',{exact:true}).waitFor();await inspect(page,'mobile-ingredient-form');
 await nav(page,'設定');await page.getByRole('combobox',{name:'設定メニュー',exact:true}).click();await page.getByRole('option',{name:'製品・校正',exact:true}).click();await page.getByRole('button',{name:'製品データを保存',exact:true}).waitFor();await inspect(page,'product-editor');
 await page.getByRole('combobox',{name:'設定メニュー',exact:true}).click();await page.getByRole('option',{name:'AI学習管理（管理者向け）',exact:true}).click();await page.getByRole('button',{name:'CPU学習を開始',exact:true}).waitFor();await inspect(page,'admin');
 console.log(JSON.stringify({passed:results.length,results,artifacts:work},null,2));
})().catch(e=>{console.error(e);process.exitCode=1;}).finally(async()=>{if(browser)await browser.close();proc.kill('SIGTERM');});
