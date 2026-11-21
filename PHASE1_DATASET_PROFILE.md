# PHASE 1 — DATASET INSPECTION REPORT (v2, corrected)

> Status: COMPLETE — corrected after independent review.
> Purpose: verify the real dataset before designing/building anything.
> Every number below was reproduced by a script/command on the real files (scripts listed in §9).

## 1. Location of raw data

- Instructions (PDF): `Catalist_Agentic_AI_Intern_Build_Test.pdf`
- Dataset root: `Catalist_AI_Intern_Build_Test_Data (1)/`
  - `transcripts/` — 138 `.txt` meeting files
  - `questions.json` — 25 questions + brief task
  - `CREDITS.txt` — AMI corpus licence (CC BY 4.0)

## 2. Verified dataset facts (all files parsed)

| Item | Verified value |
|---|---|
| Transcript files | **138** |
| Teams | **35** |
| Prefix groups | **ES: 15, IS: 10, TS: 10** |
| Headers valid | **138 / 138** |
| Total turns (parsed) | **48,148** |
| Avg turns per file | **348.9** |
| Min / max turns | **51 (IS1004a) / 1034 (TS3005d)** |
| Speakers per file | **exactly A, B, C, D in all 138 files** |
| Spoken words (body only) | **676,745** (matches PDF ~676k) |
| Words incl. headers/labels | ~826,709 |
| Teams with 4 meetings | **33** |
| Teams with 3 meetings | **2** — `IS1002` (b,c,d; a missing), `IS1005` (a,b,c; d missing) |
| Meeting-number vs letter mismatch | **NONE** |
| Non-ASCII characters | **0** (clean UTF-8) |
| Turn-header regex near-misses | **0** |
| Digits in SPOKEN text | **0** — all digits are timestamps/headers (see §3) |

## 3. CORRECTION A1 — numbers and costs

**Old report (wrong):** "Files containing digits: 138/138" and Challenge #4 said numbers "must be extracted and added deterministically".

**What the data actually shows (re-checked):**
- Counted digit characters in the **spoken text only** (header + turn-header lines removed): **0 digit characters in all 138 files**.
- All digits in the files live in the header (`Meeting number: 1`) and the turn timestamps (`00:00:13`). They are **not** spoken content.
- Prices are spoken as **number words + currency words**. Turns containing currency words (`euro(s)`, `dollar(s)`, `pound(s)`, `cent(s)`, `pence`, `quid`): **441**. (Including "point" as a decimal marker: 1,004 turns.)

**10 real examples (meeting id + speaker + exact snippet):**
1. `[ES2002a] Speaker B`: "...we're gonna be selling this remote control for twenty five Euro, um and we're aiming to make fifty million Euro...."
2. `[ES2002a] Speaker D`: "...basic product podi positioning, the twenty five Euro remote control might be a big hit in London..."
3. `[ES2002a] Speaker D`: "...is selling twenty five Euros, is that sort of the thi is this gonna to be like the premium product kinda thing..."
4. `[ES2002a] Speaker B`: "...Well twenty five Euro, I mean that's um that's about like eighteen pounds or something, isn't it?..."
5. `[ES2002b] Speaker B`: "...Well i if if like twenty five Euro is our is our selling price then you can imagine..."
6. `[ES2002b] Speaker A`: "...'Cause twenty five Euros for a remote, how much is that lo locally in pounds?..."
7. `[ES2002b] Speaker B`: "...It's about sixteen, seventeen pounds, I think...."
8. `[ES2002b] Speaker A`: "...what, ten ten quid for a remote? Like a simple replacement, right...."
9. `[ES2002d] Speaker B`: "...something has to go to the tune of two point t three Euro, so let me see... oh yes sorry, four point three...."
10. `[ES2002d] Speaker B`: "...And now we're under budget. So we do have point five Euro to play with if we wanted...."

**Corrected Challenge #4 (number handling):**
- **LLM is needed** to: turn spoken number words into values ("twenty five" -> 25, "two point three" -> 2.3, "fifty million"), decide **which number belongs to which component**, and decide whether a number is a **total**, a **limit**, a **unit price**, or a **currency conversion**.
- **Plain code can and should** do: verifying that the quote containing the number exists in the source, normalising units/currency, and doing the **arithmetic** once values are extracted. Code must never invent a number; the LLM must never be trusted to do the final arithmetic.

## 4. B1 — Citation rules (from `questions.json` -> `instructions`)

Re-read `instructions` word by word. Rules relevant to citations and answer form:

| # | Rule (verbatim intent) | Source |
|---|---|---|
| 1 | Every question needs **at least two meetings** of the named team | instructions[2] |
| 2 | Answer in **one to three plain sentences** | instructions[2] |
| 3 | Speakers are labelled by **letter A/B/C/D** exactly as in the transcript; **give the letter**, and the **role too if clear** | instructions[3] |
| 4 | **1 to 3 citations** per answer | instructions[4] |
| 5 | A citation = **meeting id + quote** | instructions[4] |
| 6 | Quote is **at least six words** | instructions[4] |
| 7 | Quote **copied exactly** from that transcript | instructions[4] |
| 8 | Quote must be **from within one speaker turn** | instructions[4] |
| 9 | Matching **ignores case and spacing** | instructions[4] |
| 10 | **Paraphrases fail** | instructions[4] |

- **Punctuation:** the instructions say only case and spacing are ignored. Punctuation is **not** stated as ignored, so the safe rule is to **keep exact punctuation**.
- **Word counting:** "at least six words" is measured on the quote string (whitespace-separated tokens).

## 5. B2 — Turn-length statistics (design-critical)

Computed over all **48,148** turns (turn = text between two turn headers):

| Metric | Value |
|---|---|
| Turns | 48,148 |
| Median words/turn | **5.0** |
| 90th percentile words/turn | **34.0** |
| Max words/turn | **736** |
| Min words/turn | **1** |
| Turns with **>= 6 words** (min quote length) | **23,821 (49.5%)** |
| Turns with **< 6 words** | **24,327 (50.5%)** |

**Design consequence:** only turns with **>= 6 words** can host a valid citation, i.e. **only ~49.5% of turns are citable**. Retrieval must therefore **index turns (with stable turn IDs)**, not only large chunks, and the agent must receive turn IDs/metadata so it can attach exact quotes. Big chunks may be used for recall, but final citations must resolve to a single turn.

## 6. B3 — TS3012 brief must exclude the final meeting

- Verified: **TS3012 has all four meetings a, b, c, d**.
- `brief_task.instruction`: "Write the brief Team TS3012 should read **before their final meeting**."
- Final meeting = `d`. Therefore the brief may use **only `TS3012a`, `TS3012b`, `TS3012c`**.
- **Rule:** retrieval for the brief must have a **hard, code-enforced `as_of` filter that excludes `TS3012d`** — not a request in the prompt.

## 7. B4 — Answer-length conflict

- **PDF page 1:** "Answer the 25 questions, each in **a sentence or two**." (= 1–2 sentences)
- **`questions.json` instructions[2]:** "Answer each one in **one to three plain sentences**." (= 1–3 sentences)
- **Safe rule that satisfies both: 1–2 sentences per answer.** (Inside 1–3, and inside "a sentence or two".)

## 8. B5 — What the 25 questions require

Tags: **CD** = changed decision (first -> final -> reason/who), **CF** = commitment follow-up, **NUM** = number/cost/price, **SPK** = speaker identification, **WM** = which-meeting, **RSN** = reason.

| Q | Team | Meetings | Tags | Evidence needed |
|---|---|---|---|---|
| Q01 | TS3006 | a,b,c,d | CD,NUM,RSN | LCD vs speech recognition: dropped vs kept + cost reason |
| Q02 | ES2003 | a,b,c,d | CD,RSN | case material first -> final + why switched |
| Q03 | TS3009 | a,b,c,d | CD,SPK | who proposed replacing titanium + what instead + kept? |
| Q04 | TS3011 | a,b,c,d | CD,NUM | combined energy source + what costs did to it |
| Q05 | TS3007 | a,b,c,d | CD,SPK | energy source first -> final + who proposed change |
| Q06 | ES2008 | a,b,c,d | CD,SPK | lost-remote feature fate + main objector + objection |
| Q07 | IS1008 | a,b,c,d | CF,NUM,SPK | who asked to cost a display + figure reported |
| Q08 | ES2007 | a,b,c,d | SPK,CD | who raised health complaint + what + design response |
| Q09 | **IS1002** | **b,c,d** | NUM,CD,SPK | first price + later range + who pushed higher price |
| Q10 | TS3003 | a,b,c,d | CD,RSN | age group chosen + why that group |
| Q11 | ES2011 | a,b,c,d | CF,SPK | PM check on requirement meaning + later answer |
| Q12 | ES2004 | a,b,c,d | CD,RSN | voice recognition fate + why + easy-to-find alternative |
| Q13 | TS3004 | a,b,c,d | CD,WM | speech recognition final? + last reversal meeting |
| Q14 | IS1009 | a,b,c,d | NUM,CF | pending-cost feature + final total + included? |
| Q15 | ES2016 | a,b,c,d | NUM,CD | over-budget total + limit + component removed |
| Q16 | TS3010 | a,b,c,d | CD,SPK,NUM | PM's preferred control + replacement + other sacrifice |
| Q17 | ES2009 | a,b,c,d | CD,WM,NUM | meeting touch screen dropped + kept feature + cost/remote |
| Q18 | ES2014 | a,b,c,d | CD,SPK | lost-remote plan first -> new + who suggested presentation |
| Q19 | IS1003 | a,b,c,d | CD,SPK | natural material -> replacement + who suggested switch |
| Q20 | ES2008 | a,b,c,d | CF,SPK | TV-only-buttons question + who would check + answer |
| Q21 | IS1007 | a,b,c,d | CF,NUM,SPK,CD | who checked prices + all-functions total + LCD decision |
| Q22 | ES2010 | a,b,c,d | CD,SPK,RSN | lithium suggester + alternative preference + final choice |
| Q23 | ES2006 | a,b,c,d | CD,NUM | kept headline feature + swapped/simplified under budget |
| Q24 | ES2002 | a,b,c,d | CD,SPK | first power source + PM-liked addition + final |
| Q25 | ES2011 | a,b,c,d | CD,RSN | fruit/vegetable shapes decision + why |

- **Questions needing more than one team: NONE.** Each question names exactly one team (the brief is a separate TS3012 task).
- **Question team with only 3 meetings: `IS1002` (Q09), meetings b,c,d** (no `a`). `IS1005` also has 3 meetings but is not used by any question.
- All other question teams have 4 meetings.

## 9. B6 — Speaker letters across meetings (tested, not assumed)

**Method:** for each selected team, extracted explicit role statements (self-introductions / direct role mentions) per meeting and compared the letter->role mapping across a,b,c,d. Also spot-checked every team flagged by a naive regex for possible change.

**In-depth check (5 teams):**

| Team | Mapping found | Consistent across meetings? |
|---|---|---|
| TS3012 | A=PM, B=UI, C=Industrial, D=Marketing (from `a`) | No restatement in b/c/d; **no contradiction** |
| ES2008 | A=PM, B=Industrial, C=UI, D=Marketing (D confirmed by "our marketing expert" in `d`) | Consistent |
| IS1002 | A=PM ("my Project Manager salary", `c`), B=Industrial (`b`), C=UI (technical role, `d`), D=Marketing (`b`,`c`,`d`) | Consistent across b,c,d |
| ES2003 | A=UI, B=PM, C=Marketing, D=Industrial (`a` intros; `b` consistent) | Consistent |
| TS3006 | A=PM (chair), D=Marketing (`a`–`d`); B/C not explicitly stated | Consistent where stated |

**Spot-check of 8 regex-flagged teams** (`TS3007, IS1008, IS1009, ES2009, ES2006, ES2013, IS1007, ES2004`): every apparent "change" was a **false positive** (topic mentions like "marketing", negations like "I'm not a very good industrial designer", or addressing another person like "I'm gonna turn over to the Industrial Designer").

**Result: no genuine letter->role change found.** Speaker letters appear stable per team across meetings. Caveat: later meetings often do **not** restate roles, so for many teams the role is only explicitly evidenced in the first available meeting. Role mapping is therefore **best-effort**; the letter is authoritative, the role is attached only when explicit.

## 10. B7 — Role-mapping numbers (reproducible, best-effort)

**Regex used (exact):**
```
(?:i am|i'm|i'll be|i will be|i'm gonna be|i'm going to be|my role is)\s+
(?:the|your|a|an)?\s*(?:(?!not\b)[a-z']+\s+){0,5}?
(project manager|industrial designer|user interface(?: designer)?|marketing(?: expert| director| manager)?)
```
Applied to each team's **first available meeting** (meeting `a`, or `b` for `IS1002`).

- **Teams with >= 1 role detected: 26 / 35.**
- **Teams with no role detected (9):** `ES2004, ES2005, ES2013, IS1001, TS3004, TS3006, TS3008, TS3010, TS3011`.
- This **differs from the old report (28/35, 7 teams)**; the count depends on the regex, exactly as the reviewer noted.
- Known limitations (why best-effort): it **misses** names with punctuation (e.g. `ES2004a` "I'm Sarah, the Project Manager") and can **false-positive** (e.g. `ES2009a` Speaker A also matched "User Interface" via "I'm gonna need you guys the User Interface Designers"). Role detection must never be treated as authoritative.

## 11. Corrections log

| Item | Old report said | Data shows | How checked |
|---|---|---|---|
| A1 | "Files containing digits: 138/138"; numbers "added deterministically" | **0 digits in spoken text**; digits only in timestamps/headers; prices spoken as words (441 currency turns) | counted digits after removing header+turn lines; grep currency words; 10 examples |
| B1 | citation rules not listed | 10 citation/answer rules extracted (6-word min, one turn, case+spacing ignored, 1–3 citations, etc.) | re-read `instructions` array |
| B2 | turn stats absent | median 5, p90 34, max 736; **23,821 (49.5%)** turns >=6 words | per-turn word count over 48,148 turns |
| B3 | brief rule absent | brief must exclude `TS3012d`; use a,b,c; code-enforced `as_of` | verified TS3012 a–d + brief_task text |
| B4 | length rule absent | PDF 1–2 sentences vs JSON 1–3 → **use 1–2** | read PDF p.1 + instructions[2] |
| B5 | classification absent | 25-row table; no Q needs >1 team; **IS1002 (Q09) has 3 meetings** | read all 25 questions + team meeting map |
| B6 | assumed roles per team (untested) | letters stable; no genuine change found; role best-effort | per-meeting role extraction + manual spot-check of 13 teams |
| B7 | 28/35, 7 teams | **26/35, 9 teams** with regex above; regex-dependent | exact regex applied to first meeting |

## 12. Design consequences for Phase 2 (from verified facts only)

- **Index turns, not only chunks:** ~50.5% of turns are <6 words and cannot host a valid quote; retrieval must return turn IDs and metadata.
- **Deterministic quote validator:** case/space-insensitive, >=6 words, **inside one turn**; reject/repair, never fabricate.
- **Numbers:** LLM converts spoken number words + maps number->component; **code** verifies quotes and does arithmetic; never trust LLM arithmetic.
- **Brief:** hard `as_of` filter that **excludes `TS3012d`** (use a,b,c).
- **Answers:** 1–2 sentences (satisfies both sources), 1–3 citations each, name speaker letter (+role only if explicit).
- **Roles:** per-team best-effort from first available meeting; letter is authoritative; handle missing `a` (IS1002) and 3-meeting teams.

## 13. Scripts used (reproducible)

All checks were run with Python 3.13 on the real files:
- `analyze_data.py` — file/team/meeting counts, word totals
- `phase1_verify.py` — 27-point scorecard over the old report
- `review_a1_b2.py` — spoken-text digit count, currency examples, turn-length stats
- `review_b6_manual.py` / `review_b6_conflicts.py` — per-meeting role statements + conflict verification
- `review_b7_b5.py` — exact role regex (26/35), question-team meeting counts

## 14. Where this is saved

`D:\Claude_projects\internship_test\PHASE1_DATASET_PROFILE.md` (summary updated in `plan.md`)
