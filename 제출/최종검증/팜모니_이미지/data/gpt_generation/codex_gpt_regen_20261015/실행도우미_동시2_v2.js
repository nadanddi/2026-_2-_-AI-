async function(start,end){
 await yield_control();
 const save=eval("("+load("regenSave")+")"),jobs=load("regenJobs"),root=load("regenRoot");
 if((load("regenViewedThrough")||0)<end)throw Error("Sources not viewed through "+end);
 for(let n=start;n<=end;n+=2){
  const batch=jobs.slice(n-1,Math.min(n+1,end));
  for(const job of batch){
   if(load("regenRecords").some(x=>x.n===job.n))throw Error("Already done "+job.n);
   await tools.apply_patch("*** Begin Patch\n*** Add File: "+root+"/사전프롬프트/"+String(job.n).padStart(3,"0")+".json\n+"+JSON.stringify({...job,prompt:load("regenPrompt")},null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
  }
  const returns=await Promise.allSettled(batch.map(job=>tools.image_gen__imagegen({prompt:load("regenPrompt"),referenced_image_paths:[job.source],transparent_background:false})));
  store("regenPairReturns",returns.map((r,i)=>({n:batch[i].n,status:r.status,output_hint:r.status==="fulfilled"?r.value.output_hint:null,error:r.status==="rejected"?String(r.reason):null})));
  let failed=false;
  for(let i=0;i<returns.length;i++){
   const r=returns[i],job=batch[i];
   if(r.status==="fulfilled"){generatedImage(r.value);await save(job,r.value);}
   else{failed=true;const f=load("regenFailures");f.push({n:job.n,file:job.source_file,error:String(r.reason)});store("regenFailures",f);text({error:String(r.reason),n:job.n});}
  }
  await yield_control();
  if(failed)break;
 }
}
