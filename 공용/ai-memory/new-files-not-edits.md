---
name: new-files-not-edits
description: "In the agri AI competition repo, never modify a file that was already produced or used; create a new file with a new name instead"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 0f62f980-d110-4c91-abd7-dffbf0ff506c
  modified: 2026-09-25T17:23:11.716Z
---

Once a script, submission file, manifest, or package has been created or used for a submission, do not edit or overwrite it. Make a new version under a new name (e.g. make_submission_v5.py, build_package_v5.py, submission_05.csv, *_05.zip).

**Why:** the user objected when I extended build_package.py in place to support a new submission and when a verification step regenerated submission_04.csv in place right after it had been uploaded. Uploaded submissions and their reproduction ZIPs must stay traceable to the exact files that produced them (competition rule 8.2: reproduction failure voids the score).

**How to apply:** before any Edit/Write/overwrite of experiment code or results (집/*/research, 집/코덱스/analysis, 연구실/*) or anything in 제출/, check whether the target already exists and was used; if so, copy to a new versioned name and change the copy. Run package verifications only inside an extracted copy, never in the source tree. Related: [[agri-competition-submission-flow]]
