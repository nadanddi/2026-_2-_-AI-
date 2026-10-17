---
name: image-approach-official-types
description: "Image mission approach the user set (2026-10-10) — augment external real photos with the official 6 manipulation types and train; don't treat AI-Hub as the organizers' source; red-LED photos stay excluded"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 72cf93e1-8c65-4dfd-8614-7baba25d7126
  modified: 2026-10-10T13:46:10.945Z
---

User (2026-10-10, frustrated): the problem statement already defines how fakes were made (6 types). The method is simply: take external public real photos (AI-Hub strawberry photos are just similar-looking external photos, NOT confirmed organizer data) → apply the officially described manipulations → train the classifier.

**Why:** I kept proposing to reverse-engineer organizer fakes against AI-Hub originals and kept pushing to re-include red-LED photos after the user decided to exclude them (sample has none). That ignored the user's decision and over-assumed AI-Hub = organizer source.

**How to apply:**
- Build fakes by following the official type definitions (filter: blur/gaussian noise/salt-pepper/JPEG; composite; full/partial style; partial/full generation) applied to external real photos; don't make "match organizer pipeline via AI-Hub originals" a priority step.
- Red-LED (all-red) photos: excluded, decision final. Report diagnostics once if asked, but don't re-litigate.
- Don't state AI-Hub is the eval source; call it external similar photos.
- Related: [[lab-claude-image-mission]], [[no-exaggeration]].
