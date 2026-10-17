// functions.exec orchestration snapshot; run within tools runtime.

async function(item,r) {
 const match=String(r.output_hint||"").match(/as (C:\\[^\r\n]+?\.png) by default/);
 if(!match) throw new Error("Generated output path not found");
 const raw=match[1], root=load("rootV3"), dest=root+"/"+item.relative;
 const q=s=>"'"+s.replace(/'/g,"''")+"'";
 const cmd="$ErrorActionPreference='Stop'\n$dest="+q(dest)+"\nif(Test-Path -LiteralPath $dest){throw 'Destination exists'}\nNew-Item -ItemType Directory -Path (Split-Path -Parent $dest) -Force | Out-Null\nCopy-Item -LiteralPath "+q(raw)+" -Destination $dest\n$h1=(Get-FileHash -LiteralPath "+q(raw)+" -Algorithm SHA256).Hash\n$h2=(Get-FileHash -LiteralPath $dest -Algorithm SHA256).Hash\nif($h1 -ne $h2){throw 'Raw copy mismatch'}\n[pscustomobject]@{sha256=$h2;bytes=(Get-Item -LiteralPath $dest).Length}|ConvertTo-Json -Compress";
 const copy=await tools.exec_command({cmd,max_output_tokens:400});
 if(copy.exit_code!==0)throw new Error(copy.output);
 const proof=JSON.parse(copy.output.trim()),rec={...item,tool:"GPT_builtin_imagegen",result:dest,generated_path:raw,...proof};
 const records=load("recordsV3");records.push(rec);store("recordsV3",records);
 await tools.apply_patch("*** Begin Patch\n*** Add File: "+root+"/기록/"+item.id+"_"+item.stage+".json\n+"+JSON.stringify(rec,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
 text({calls:records.length,id:item.id,stage:item.stage});
}

async function(start,end,stage) {
 const save=eval("("+load("saveFnV3")+")"),plan=load("planV3"),root=load("rootV3");
 for(let n=start;n<=end;n++){
 const job=plan[n-1],isIntermediate=stage==="stage1";
 const input=isIntermediate?load("srcV3")+"/"+job.source:job.method==="A"?root+"/중간_GPT/"+job.id+"_A_stage1.png":load("srcV3")+"/"+job.source;
 if(!isIntermediate && !job.prompt)throw new Error("Missing concrete prompt "+n);
 const item={...job,stage,prompt:isIntermediate?job.stage1prompt:job.prompt,input,relative:isIntermediate?"중간_GPT/"+job.id+"_A_stage1.png":"결과_GPT/"+job.type+"/"+job.id+"_"+job.method+"_final.png"};
 await tools.apply_patch("*** Begin Patch\n*** Add File: "+root+"/사전프롬프트/"+item.id+"_"+stage+".json\n+"+JSON.stringify(item,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
 try{
 const r=await tools.image_gen__imagegen({prompt:item.prompt,referenced_image_paths:[input],transparent_background:false});
 generatedImage(r);await save(item,r);
 }catch(e){const f=load("failuresV3");f.push({id:job.id,stage,error:String(e)});store("failuresV3",f);text({failure:job.id,stage,error:String(e)});break;}
 await yield_control();
 }
}
