---
name: gpu-job-mailbox
description: "Heavy GPU experiments go to the Google Drive \"gpu_jobs\" mailbox; lab Codex runs them when the user says \"gpu_jobs 확인해봐\""
metadata:
  node_type: memory
  type: project
  originSessionId: 0241a1c7-a2b9-4fbe-a187-c5f98b062c2a
  modified: 2026-10-14T11:37:43.979Z
---

On 2026-10-14 the user set up a GPU job mailbox, because the home laptop has a small GPU and limited CPU and RAM.

- **Mailbox:** `G:\내 드라이브\farmai_sync\gpu_jobs\`. Google Drive for desktop syncs it automatically, so no git commit is needed.
  - `README_코덱스_GPU작업우편함.md`: the protocol for lab Codex.
  - `QUEUE.md`: the job table, with status 대기 → 진행 → 완료/오류.
  - `JOBnnn_<name>\JOB.md` plus a bundle ZIP; results come back in `results\` with `DONE.txt` or `ERROR.txt`.
- **Who runs it:** the user tells lab Codex "gpu_jobs 확인해봐", and Codex runs the queued job exactly as written. Lab Claude is near its usage limit, so the lab side is Codex.
- **How to apply:**
  - When a pre-registered experiment is too heavy for the laptop (e.g. many folds × TabPFN × R3), build a self-contained bundle with repo-relative paths.
    - Trace the opened files with an audit hook.
    - Include the TabPFN v2 ckpt from `%APPDATA%\tabpfn`.
    - Include any existing checkpoints.
  - Test-extract the bundle into an empty dir with a stub CLAUDE.md before sending it.
  - Add the job to QUEUE.md as the last step of the turn, and tell the user to ping Codex.
  - Interpreting, judging and cataloguing the results stays with home Claude.
- **Limits:**
  - SSH to the lab server ADILAB (port 10079) timed out from home on 2026-10-14, so external access looks blocked.
  - Creating SSH keys was denied by the permission classifier.
  - Never use or store the server password.

First job: JOB001, SZ2 season-index removal (66 folds). Related: [[explore-autonomously]].
