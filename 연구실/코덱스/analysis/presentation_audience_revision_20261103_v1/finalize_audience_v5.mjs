import path from 'node:path';
import {pathToFileURL} from 'node:url';
const skill='C:/Users/0317/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const workspaceDir='C:/work/farmai/연구실/코덱스/analysis/presentation_audience_revision_20261103_v1';
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
try {
 const r=await finalizePresentation({workspaceDir,candidatePath:path.join(workspaceDir,'팜모니AI_발표자료_청자이해보완_v8.pptx'),finalPath:path.join(workspaceDir,'팜모니AI_발표자료_청자이해보완_검증본_v1.pptx'),pythonExecutable:'C:/Users/0317/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),explicitTotalSlideCount:33,layoutArgs:['--expected-slide-size-emu','12192000,6858000'],verifyArtifactToolImport:true,receiptPath:path.join(workspaceDir,'finalizer_receipt_v5.json')});
 console.log(JSON.stringify(r));
} catch(e) { console.error(e.stack);process.exitCode=1; }
