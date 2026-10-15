---
name: lab-gpu-server-rules
description: "Shared lab GPU server ADILAB.iptime.org (2× RTX 6000 Ada 48GB) — mandatory usage rules from the lab admin; user's env is `nadan`"
metadata:
  node_type: memory
  type: reference
  originSessionId: 9d690533-e070-4e3e-b7a9-4781b4bcbd43
  modified: 2026-10-12T01:32:28.486Z
---

Server: `ADILAB.iptime.org` (ssh config: user adilab, port 10079), shared by the lab. 2× NVIDIA RTX 6000 Ada (48GB each). User's work dir `~/storage/nadan`, conda env `nadan` (Python 3.10). Auth is password-based; Claude must not type/use the password — the user registers an SSH key themselves (seen 2026-10-12).

Mandatory rules (admin note "기수형멋져요.txt", user said 반드시 준수):
- Always `conda activate nadan` before anything.
- Install packages only with `python -m pip install <pkg>` (inside the env).
- New env only via `conda create -n <name> python=<ver>`.
- Before running any model: run `nvidia-smi` and use only a GPU nobody else is using; pin it explicitly (torch `cuda:N` / `CUDA_VISIBLE_DEVICES=N`).
- In VS Code: double-check the selected interpreter/kernel is the user's env.

Related: [[lab-claude-image-mission]].

Access notes (2026-10-15): reachable only from the lab PC (same iptime router; port 10079 closed from outside/home). From git-bash use `/c/Windows/System32/OpenSSH/ssh.exe` (git's ssh fails host-key check); from PowerShell plain `ssh` works but embedded double quotes/pipes in the remote command get mangled — upload a .sh with scp and run `bash file.sh`. Lab Codex may also run jobs on GPU0.
