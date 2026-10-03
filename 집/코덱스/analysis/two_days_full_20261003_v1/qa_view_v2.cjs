const {chromium}=require('C:/Users/aozks/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const {pathToFileURL}=require('node:url');
const fs=require('node:fs/promises');
const path=require('node:path');
async function main(){
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1450,height:1000},deviceScaleFactor:1});
    const errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(pathToFileURL(path.join(__dirname,'전체비교표_v2.html')).href);
    if(errors.length)throw new Error(errors.join(' / '));
    const count=()=>page.locator('details:visible').count();
    if(await count()!==27)throw new Error('27 panels required');
    await page.screenshot({path:path.join(__dirname,'preview_summary_v2.png')});
    await page.locator('#expand').click();if(await page.locator('details[open]').count()!==27)throw new Error('expand failed');
    await page.locator('#collapse').click();if(await page.locator('details[open]').count()!==0)throw new Error('collapse failed');
    await page.locator('#group').selectOption('입력');if(await count()!==19)throw new Error('input filter failed');
    await page.locator('#diff').check();if(await count()!==8)throw new Error('different filter failed');
    await page.locator('#diff').uncheck();await page.locator('#group').selectOption('all');
    await page.locator('a[href="#in_temp"]').click();
    if(await page.locator('#in_temp').getAttribute('open')===null)throw new Error('anchor did not open');
    await page.locator('#in_temp').screenshot({path:path.join(__dirname,'preview_in_temp_v2.png')});
    if(errors.length)throw new Error(errors.join('\n'));
    await fs.writeFile(path.join(__dirname,'view_verification.json'),JSON.stringify({status:'PASS',panels:27,inputs:19,different_inputs:8,expand_collapse:true,anchor_open:true,javascript_errors:errors,engine:'Microsoft Edge headless',screenshots:['preview_summary_v2.png','preview_in_temp_v2.png']},null,2));
    console.log('VIEW_VERIFICATION_PASS');
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
