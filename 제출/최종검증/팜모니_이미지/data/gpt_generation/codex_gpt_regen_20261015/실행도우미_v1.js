const save = async function(job,result){
 const root=load("regenRoot"),q=s=>"'"+s.replace(/'/g,"''")+"'",hint=result.output_hint||"";
 const match=hint.match(/as (C:\\[^\r\n]+?\.png) by default/);
 if(!match)throw Error("Generated PNG path not found");
 const raw=match[1],dest=root+"/"+job.result_file;
 store("regenPending",{job,raw,dest});
 const saved=await tools.exec_command({cmd:
 "$ErrorActionPreference='Stop'\n$taskRaw="+q(raw)+"\n$taskDest="+q(dest)+"\n"+
 "if(Test-Path -LiteralPath $taskDest){throw 'Result exists'}\nCopy-Item -LiteralPath $taskRaw -Destination $taskDest\n"+
 "$taskRawHash=(Get-FileHash -LiteralPath $taskRaw -Algorithm SHA256).Hash\n$taskHash=(Get-FileHash -LiteralPath $taskDest -Algorithm SHA256).Hash\nif($taskRawHash -ne $taskHash){throw 'Copy hash mismatch'}\n"+
 "$taskMapping=[pscustomobject][ordered]@{原本="+q(job.source_file)+";結果="+q(job.result_file)+"}\n"+
 "$taskCsv=[pscustomobject][ordered]@{'원본파일명'="+q(job.source_file)+";'결과파일명'="+q(job.result_file)+"}\n"+
 "$taskCsv | Export-Csv -LiteralPath "+q(root+"/목록.csv")+" -NoTypeInformation -Encoding utf8BOM -Append\n"+
 "[pscustomobject]@{raw="+q(raw)+";result=$taskDest;sha256=$taskHash;bytes=(Get-Item -LiteralPath $taskDest).Length;completed_at=(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK')} | ConvertTo-Json -Compress",max_output_tokens:1000});
 if(saved.exit_code!==0)throw Error(saved.output);
 const record={...job,prompt:load("regenPrompt"),tool:"GPT_builtin_imagegen",...JSON.parse(saved.output)};
 const records=load("regenRecords");records.push(record);store("regenRecords",records);
 const id=String(job.n).padStart(3,"0");
 await tools.apply_patch("*** Begin Patch\n*** Add File: "+root+"/기록/"+id+".json\n+"+JSON.stringify(record,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
 store("regenPending",null);text({completed:records.length,n:job.n,file:job.result_file});
};
const run = async function(start,end){
 const save=eval("("+load("regenSave")+")"),jobs=load("regenJobs"),root=load("regenRoot");
 for(let n=start;n<=end;n++){
  const job=jobs[n-1];if(load("regenRecords").some(x=>x.n===n))throw Error("Already done "+n);
  await tools.apply_patch("*** Begin Patch\n*** Add File: "+root+"/사전프롬프트/"+String(n).padStart(3,"0")+".json\n+"+JSON.stringify({...job,prompt:load("regenPrompt")},null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
  try{const r=await tools.image_gen__imagegen({prompt:load("regenPrompt"),referenced_image_paths:[job.source],transparent_background:false});generatedImage(r);await save(job,r);}
  catch(e){const failures=load("regenFailures");failures.push({n,file:job.source_file,error:String(e)});store("regenFailures",failures);text({error:String(e),n});break;}
  await yield_control();
 }
};
