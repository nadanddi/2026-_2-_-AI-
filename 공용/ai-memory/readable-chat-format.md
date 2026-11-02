---
name: readable-chat-format
description: "In chat, avoid wide/merged markdown tables for overviews; use short grouped lists (one line per item) instead"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d4d3c4ad-0722-4325-9f75-bd37cce2822c
  modified: 2026-11-02T16:38:09.827Z
---

The user (2026-11-03) said the table I wrote in chat for the presentation order was "가독성이 떨어져". That table had section-header rows mixed into the data rows, empty cells, and long sentences inside cells. What they wanted was a readable overview, not a separate script document.

**Why:** In the chat pane, markdown tables with merged-looking group rows, `| |` gaps and long cell text are hard to scan.

**How to apply:**
- For orders, plans and summaries in chat, group items under short bold headings (e.g. per part).
- Give one line per item: number, name, time, then a short gist.
- Use a table only for truly columnar data (few columns, short cells, no group rows).
- Don't build a new file or page when the user only asked for a readable answer.

Related: [[plain-explanations]].
