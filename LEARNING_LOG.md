# Learning Log

Short, dated notes on what I learned, what broke, and what I changed.

---

## 2026-10-04 — Stage 1: Loading the OFAC SDN list

**What I learned**
- OFAC (U.S. Treasury) publishes the SDN list: people, companies, vessels and aircraft that U.S. persons and banks can't do business with. Every U.S. bank must screen customers and payments against it.
- The current list has **19,488 entries and 20,236 aliases → 39,467 names to screen against**. There are more aliases than primary names, because sanctioned parties deliberately use alternate spellings. Exact-match screening alone will miss them.
- Russia-related designations (RUSSIA-EO14024, 5,664 entries) make up roughly **29%** of the list, far ahead of terrorism (SDGT, 2,185) and narcotics (SDNTK, 1,332).

**What broke and how I fixed it**
1. **Blank values weren't cleaned.** OFAC writes blanks as `-0- ` *with a trailing space*. My code replaced `-0-` before stripping spaces, so nothing matched, and companies were labelled `-0-` instead of `entity`. Fix: strip whitespace first, then replace. *Lesson: inspect the raw file; real data has invisible quirks.*
2. **Apostrophes split names.** `Sa'id` became `sa id` (two tokens), which would hurt matching later. Fix: delete apostrophes before turning other punctuation into spaces. *Lesson: normalization choices directly decide whether a sanctioned person gets caught.*

**Decision**
- Only individuals get flipped from `LAST, First` to `first last`. Company names like `ANGLO-CARIBBEAN CO., LTD.` also contain commas, and flipping them would scramble them.

---

## 2026-10-05 — Stage 2: Building a fake bank

**What I built**
- 500 synthetic customers, ~4,600 transactions (Jul–Sep 2026), and an answer key.
- Planted 10 sanctions hits using real SDN names in five forms: exact, misspelled, reordered, missing middle name, alias.
- Planted 5 false-positive traps: innocent people who share a surname with someone on the list.
- Planted 5 suspicious behaviours (3 customers each): structuring, rapid movement, high-risk geography, velocity spike, round amounts.

**Why an answer key matters**
- Without one, I can't measure anything. With it, I can report how many real hits my screener catches and how many innocent customers it wrongly flags. This is the same tradeoff real fincrime teams tune every day.

**Notes / open questions**
- _(add what you notice when you open the CSVs)_
