# Custom Multi-Meeting Evaluation — Human Review Scoring Sheet

> **Status:** `manual_rubric_score: PENDING_HUMAN_REVIEW` (explicitly unpopulated in `eval/eval_report.json` as `null`).
> **Policy:** Automated checks (`ok`, citation validity, gold retrieval recall, `required_facts` substring recall) and transcript-grounded auditor observations are provided below to aid review, **never** as fabricated human rubric scores. A human evaluator must inspect each pair and assign the final `0 / 1 / 2` rubric score.

---

## 1. Scoring Rubric (0–2 Scale per Question)

| Score | Meaning | Criteria |
|---|---|---|
| **2** | **Correct** | Accurately answers every sub-part of the multi-meeting question, consistent with raw transcripts, with 1–3 verbatim verified citations supporting the claims. |
| **1** | **Partially Correct** | Answers at least one core sub-part accurately with valid citations, but omits a secondary detail, is imprecise on one element, or cites only a single meeting when multiple meetings carry key evidence. |
| **0** | **Incorrect / Failed** | Contradicts the transcripts on a core fact (wrong decision, speaker, or number), answers an unrelated point, or failed to generate an answer (`AllProvidersFailed`). |

---

## 2. Automated Summary (`eval/eval_report.json`)

- **Custom question set:** `eval/eval_questions.json` (`12` manually authored multi-meeting questions across `12` AMI teams **completely disjoint** from the 25 original questions: `ES2005, ES2012, ES2013, ES2015, IS1000, IS1001, IS1004, IS1005, IS1006, TS3005, TS3008, TS3012`). All `expected_answer`, `gold_turn_ids`, and `required_facts` were authored by directly reading raw `.txt` transcripts (never copied from agent output).
- **Live execution outcome:**
  - **Passed deterministic validation (`ok=True`):** `10 / 12` (`83.33%`)
  - **Failed (`llm_error:AllProvidersFailed`, free-tier quota exhausted; 0 fabricated answers):** `2 / 12` (`C08`, `C12`)
  - **Citation validity (verified verbatim single-turn quotes):** `22 / 22` (`100.0%`)
  - **Multi-meeting citation rate (`>=2` distinct meetings):** `7 / 12` (`58.33%` overall; `7 / 10` = `70.0%` of live-completed)
  - **Mean gold retrieval recall (agent `retrieved` vs manual `gold_turn_ids`):** `0.5764` overall (`0.6917` across the 10 live-completed questions)
  - **Mean gold citation recall:** `0.2583` overall (`0.3100` across the 10 live-completed questions)
  - **Mean deterministic `required_facts` recall:** `0.6320` overall (`0.7583` across the 10 live-completed questions)
  - **Automated fact tiers:** `4` all-facts-matched (`C01, C03, C06, C09`), `6` partial-facts-matched (`C02, C04, C05, C07, C10, C11`), `2` failed/unanswered (`C08, C12`).
- **No-team-filter retrieval benchmark (all `12 / 12` manual-gold questions, 0 LLM calls):**
  - `k=8`: team-filtered recall `0.1667` -> no-team recall `0.0208` (gap `+0.1459`, `94.79%` cross-team contamination)
  - `k=16`: team-filtered recall `0.3167` -> no-team recall `0.0958` (gap `+0.2209`, `90.62%` cross-team contamination)
  - Guardrail verified: `RunContext(team=None, mode='answer'|'brief')` raises `ValueError`.
- **Independent transcript-check breakdown (preliminary audit — NOT human rubric):**
  - Factually accurate across core points: **5** (`C01`, `C03`, `C04`, `C06`, `C09`)
  - Partially accurate: **3** (`C02`, `C07`, `C11`)
  - Factually incorrect: **2** (`C05`, `C10`)
  - Failed (provider exhaustion, no answer fabricated): **2** (`C08`, `C12`)

---

## 3. Per-Question Review Table (Human Grader Sign-Off)

| QID | Team | Model | Validated (`ok`) | Cites Valid | Meetings Cited | Gold Ret. Recall | Auto Fact Recall | Auditor Transcript Note (Not Final Score) | Human Score (`0/1/2`) | Human Reviewer Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| **C01** | `ES2005` | `qwen/qwen3.8-27b` | `True` | `2/2` | `ES2005d` (`single_meeting`) | `1.00` | `3/3 (1.00)` | Factually exact (`14.6` -> double-curved entered as qty 2 -> `11.60`); single-meeting citation (`d` only). | `PENDING_HUMAN_REVIEW` | |
| **C02** | `ES2012` | `qwen/qwen3.8-27b` | `True` | `3/3` | `ES2012c, ES2012d` | `1.00` | `2/4 (0.50)` | Partial: got Speaker C battery-in-cover + Speaker A connector/soldered objection right; missed meeting `d` push-buttons/slider & `12.50` Euro total. | `PENDING_HUMAN_REVIEW` | |
| **C03** | `ES2013` | `qwen/qwen3.8-27b` | `True` | `2/2` | `ES2013c, ES2013d` | `0.50` | `5/5 (1.00)` | Factually exact (titanium too expensive/hard to double-curve; plastic+rubber; `9.20` Pounds). | `PENDING_HUMAN_REVIEW` | |
| **C04** | `ES2015` | `qwen/qwen3.8-27b` | `True` | `2/2` | `ES2015c, ES2015d` | `0.83` | `3/4 (0.75)` | Factually exact (Speaker B ID, kinetic + anti-RSI rubber survived into `d`, `12.20` Euros — auto-fact missed only phrasing `"twelve point twenty"` vs `"twelve twenty"`). | `PENDING_HUMAN_REVIEW` | |
| **C05** | `IS1000` | `gemini-flash-lite-latest` | `True` | `2/2` | `IS1000b, IS1000d` | `0.80` | `3/4 (0.75)` | **Incorrect:** got LCD cut right, but wrongly claimed speech rec was kept and compensated with a protective cover (`IS1000d#00120`); transcripts (`#00716–#00722`) cut both & added 2 extra buttons. | `PENDING_HUMAN_REVIEW` | |
| **C06** | `IS1001` | `qwen/qwen3.8-27b` | `True` | `2/2` | `IS1001c, IS1001d` | `1.00` | `3/3 (1.00)` | Factually exact (banana shape in `c`, yellow prevents losing it, `10.7` Euros in `d`). | `PENDING_HUMAN_REVIEW` | |
| **C07** | `IS1004` | `openai/gpt-oss-20b` | `True` | `2/2` | `IS1004d` (`single_meeting`) | `0.80` | `2/3 (0.67)` | Partial: got sample speaker dropped & rubber->plastic right, but attributed speaker removal solely to cost rather than beep-only locator (`IS1004d#00377–#00383`); `single_meeting`. | `PENDING_HUMAN_REVIEW` | |
| **C08** | `IS1005` | `None` | `False` | `0/0` | `[]` | `0.00` | `0/4 (0.00)` | **Failed (`AllProvidersFailed`):** all 7 free-tier models exhausted/cooling down; no answer fabricated. | `PENDING_HUMAN_REVIEW` | |
| **C09** | `IS1006` | `qwen/qwen3.8-27b` | `True` | `3/3` | `IS1006c, IS1006d` | `0.40` | `4/4 (1.00)` | Factually exact (Speaker C proposed titanium + push-buttons for speech budget in `c`; `d` prototype included LCD + scroll wheel). | `PENDING_HUMAN_REVIEW` | |
| **C10** | `TS3005` | `gemini-flash-lite-latest` | `True` | `2/2` | `TS3005d` (`single_meeting`) | `0.33` | `1/4 (0.25)` | **Incorrect:** got dropping sample speaker (`#00627`), but missed Speaker C dropping scroll wheel for touch-screen-only to reach `11.50` Euros (`#00654–#00661`), citing wood->brown instead. | `PENDING_HUMAN_REVIEW` | |
| **C11** | `TS3008` | `qwen/qwen3.8-27b` | `True` | `2/2` | `TS3008c, TS3008d` | `0.25` | `2/3 (0.67)` | Partial: accurately captured missing docking station & impossible 30-cent margin in `d`; in `c`, cited simplicity/overdone rather than fragility/sturdy throwable case (`TS3008c#00214–#00215`). | `PENDING_HUMAN_REVIEW` | |
| **C12** | `TS3012` | `None` | `False` | `0/0` | `[]` | `0.00` | `0/4 (0.00)` | **Failed (`AllProvidersFailed`):** all 7 free-tier models exhausted/cooling down; no answer fabricated. | `PENDING_HUMAN_REVIEW` | |

---

## 4. Detailed Side-by-Side Evidence (Expected vs. Predicted + Transcript Ground Truth)

### C01 (`ES2005`) — `ok=True`, `qwen/qwen3.8-27b`
- **Question:** Team ES2005's first production costing in meeting d came out over their 12.50 Euro limit. What initial cost did Speaker C report, why was it inflated, and what final cost did they reach once that error was corrected?
- **Manual Gold (`ES2005a#00112, ES2005d#00071, ES2005d#00083, ES2005d#00105, ES2005d#00112`):** Against their 12.50 Euro production target (ES2005a), Speaker C initially reported a cost of 14.6 Euros in meeting d because double-curved casing had been mistakenly entered as quantity two instead of one; fixing it to one removed 3 Euros and brought the total down to 11.60 Euros.
- **Predicted:** Speaker C initially reported a cost of fourteen point six Euros, which was inflated because the spreadsheet accidentally counted the double-curved shape as two separate items instead of one. Once this error was corrected by removing the duplicate entry, the final cost reached eleven Euros sixty.
- **Verified Citations:**
  - `[ES2005d | ES2005d#00071]`: `"minus the extra scroll buttons, uh it's gonna cost us fourteen point six Euros"`
  - `[ES2005d | ES2005d#00083]`: `"W why is the uh double curved two of them?"`

### C02 (`ES2012`) — `ok=True`, `qwen/qwen3.8-27b`
- **Question:** In Team ES2012, who proposed putting the batteries inside interchangeable covers, why did Speaker A object to that idea, and what final interface and production cost did Speaker A report in meeting d?
- **Manual Gold (`ES2012c#00159, ES2012c#00164, ES2012d#00057`):** Speaker C (User Interface Designer) proposed putting the batteries inside the cover in meeting c, and Speaker A objected that adding a connector would introduce unreliability compared to soldering the unit together. In meeting d, Speaker A reported a design using push buttons and a simple slider (no scroll wheel) at 12.50 Euros.
- **Predicted:** Speaker C proposed putting the batteries in the interchangeable covers, but Speaker A objected because it would introduce a connection point and unreliability compared to a completely soldered, plumbed-in design. In meeting d, Speaker A reported that the team would use a regular standard chip to meet the constraints.
- **Verified Citations:**
  - `[ES2012c | ES2012c#00159]`: `"What if we included the batteries in the cover?"`
  - `[ES2012c | ES2012c#00164]`: `"you've then got the connection and you've immediately got a sort of unreliability"`
  - `[ES2012d | ES2012d#00057]`: `"we will have to find a s a regular standard chip to to do this with"`

### C03 (`ES2013`) — `ok=True`, `qwen/qwen3.8-27b`
- **Question:** Why did Team ES2013 eliminate titanium for the remote's case in meeting c, what two case materials did they combine in the final costed design, and what total production cost did Speaker A calculate?
- **Manual Gold (`ES2013c#00220, ES2013c#00224, ES2013d#00232, ES2013d#00234`):** Speaker A explained in meeting c that titanium was eliminated because it was too expensive and could not support double curves. In meeting d, they combined plastic and rubber for the case and Speaker A calculated a total cost of 9.20 (nine Pounds twenty), well below the 12.50 limit.
- **Predicted:** Team ES2013 eliminated titanium in meeting c because it was considered too expensive, too futuristic, and difficult to shape into double curves. In the final costed design, they combined plastic and rubber for the case, and Speaker A calculated a total production cost of nine pounds twenty.
- **Verified Citations:**
  - `[ES2013c | ES2013c#00129]`: `"titanium would be great, but I think it might be a bit too expensive and too kinda spacey"`
  - `[ES2013d | ES2013d#00234]`: `"the total cost has been calculated as nine Pounds twenty out of the twelve and a half we were allowed"`

### C04 (`ES2015`) — `ok=True`, `qwen/qwen3.8-27b`
- **Question:** Which Team ES2015 member recommended kinetic energy and a rubber casing for anti-RSI comfort in meeting c, and did both features survive into meeting d's final costed design and at what total operating cost?
- **Manual Gold (`ES2015c#00021, ES2015c#00031, ES2015d#00483, ES2015d#00488, ES2015d#00496, ES2015d#00501`):** Speaker B (Industrial Designer) recommended kinetic energy and a rubber casing for anti-RSI comfort in meeting c. Both kinetic energy and the plastic-and-rubber casing remained in the final costed design in meeting d, with Speaker A reporting an operating cost of 12.20 Euros.
- **Predicted:** Speaker B (the Industrial Designer) recommended kinetic energy and a rubber casing for anti-RSI comfort in meeting c, and both features survived into meeting d's final design at a total operating cost of twelve point twenty Euros.
- **Verified Citations:**
  - `[ES2015c | ES2015c#00031]`: `"Rubber casing I thing would probably the best, if we're going for the anti-R_S_I_"`
  - `[ES2015d | ES2015d#00501]`: `"So um our operating cost is twelve twenty Euros."`

### C05 (`IS1000`) — `ok=True`, `gemini-flash-lite-latest` *(Genuine Wrong Answer — see Failure Analysis)*
- **Question:** Team IS1000 debated including both an LCD screen and speech recognition under their 12.50 Euro budget. What did they decide about both features in meeting d, and how did they compensate for cutting the LCD screen?
- **Manual Gold (`IS1000c#00161, IS1000d#00683, IS1000d#00716, IS1000d#00719, IS1000d#00722`):** Although Team IS1000 considered both features in meeting c and early meeting d, they cut both the LCD screen and speech recognition to stay within the 12.50 Euro budget. To compensate for dropping the LCD screen, they added two extra buttons.
- **Predicted:** The team decided to cut the LCD screen because it was too expensive at sixteen Euros, but they kept speech recognition and reverted to a simpler version on budget at twelve Euros. They compensated for cutting the LCD screen by adding a protective cover to the device.
- **Verified Citations:**
  - `[IS1000b | IS1000b#00325]`: `"If we're getting into universal remote territory, we're getting to L_C_D_ screens and things like that which would drive the cost up a lot."`
  - `[IS1000d | IS1000d#00120]`: `"this gives a protection to the L_C_D_ actually, giving a cover to that actually."`

### C06 (`IS1001`) — `ok=True`, `qwen/qwen3.8-27b`
- **Question:** Which fruit shape did Team IS1001 choose for their remote in meeting c, what practical advantage did Speaker A note for that choice, and what final production cost did Speaker A report in meeting d?
- **Manual Gold (`IS1001c#00174, IS1001c#00178, IS1001d#00015, IS1001d#00017`):** Team IS1001 chose a yellow banana shape in meeting c, with Speaker A noting that because a banana is yellow it helps prevent losing the remote. In meeting d, Speaker A reported a final cost of 10.7 Euros, safely below the 12.50 Euro limit.
- **Predicted:** Team IS1001 chose a banana shape for their remote in meeting c, with Speaker A noting that its yellow color makes it easy to find so users cannot lose it. In meeting d, Speaker A reported the final production cost ended up being ten point seven Euros.
- **Verified Citations:**
  - `[IS1001c | IS1001c#00178]`: `"Banana is also yellow so you you can't lost your remote control then."`
  - `[IS1001d | IS1001d#00015]`: `"And the cost ended to be ten point seven Euros."`

### C07 (`IS1004`) — `ok=True`, `openai/gpt-oss-20b`
- **Question:** During Team IS1004's final costing in meeting d, what happened to the sample speaker and the rubber casing that had been considered, and why was each removed or changed?
- **Manual Gold (`IS1004c#00220, IS1004d#00377, IS1004d#00383, IS1004d#00407, IS1004d#00414`):** Speaker A removed the sample speaker because Speaker B clarified they were only using a simple beep locator rather than voice recording. When the costing still came out over budget, Speakers B and C switched the case material from rubber to plastic so the cost would fit the budget and leave margin for the base station.
- **Predicted:** In the final costing meeting, the sample speaker was dropped because its cost was too high, and the rubber casing was replaced with plastic to reduce expenses.
- **Verified Citations:**
  - `[IS1004d | IS1004d#00376]`: `"And we have sample speaker. Yeah. Yeah. the cost of that is very high."`
  - `[IS1004d | IS1004d#00409]`: `"Well no didn't we say we wanted to do a rubber"`

### C08 (`IS1005`) — `ok=False`, `llm_error:AllProvidersFailed`
- **Question:** In Team IS1005, which signal transmission technology did Speaker B recommend instead of radio frequency in meeting b and why, and what power combination did Speaker B say was required in meeting c to support both an LCD and speech recognition?
- **Manual Gold (`IS1005b#00080, IS1005b#00082, IS1005c#00102, IS1005c#00105`):** In meeting b, Speaker B recommended sticking with infrared rather than radio frequency because infrared was much cheaper (radio frequency being 3 to 5 times more expensive) while offering fair range. In meeting c, Speaker B said combining common AA batteries with an additional solar cell was required to power both an LCD screen and speech recognition.
- **Predicted:** *(None — failed with `llm_error:AllProvidersFailed` after exhausting free-tier daily limits; no answer fabricated.)*

### C09 (`IS1006`) — `ok=True`, `qwen/qwen3.8-27b`
- **Question:** In meeting c of Team IS1006, what case material and interface did Speaker C propose and why, and how did the prototype presented in meeting d differ with respect to the display?
- **Manual Gold (`IS1006c#00021, IS1006c#00036, IS1006d#00057, IS1006d#00233, IS1006d#00410`):** In meeting c, Speaker C proposed a titanium case (instead of rubber, which was expensive and hard to mold) and simple push buttons instead of an LCD to save money for speech recognition. In meeting d, however, the prototype still used a titanium case but actually included a non-touch LCD screen navigated with two directional buttons and a jog dial/wheel.
- **Predicted:** In meeting c, Speaker C proposed a titanium case and an interface combining push buttons with speech recognition to reduce costs and free up budget for the speech technology. In meeting d, the prototype differed by including an LCD display (specifically with a backlight) alongside the push buttons and a scroll wheel, whereas the earlier discussion had considered dropping the LCD in favor of buttons and speech.
- **Verified Citations:**
  - `[IS1006c | IS1006c#00036]`: `"We'll use mainly for titanium rubber is expensive and also it's bit difficult to do all the shapes"`
  - `[IS1006c | IS1006c#00036]`: `"we want to reduce price on this technology and so that we can have enough space or enough money for"`
  - `[IS1006d | IS1006d#00410]`: `"One scroll wheel. One L_C_D_ displayed."`

### C10 (`TS3005`) — `ok=True`, `gemini-flash-lite-latest` *(Genuine Wrong Answer — see Failure Analysis)*
- **Question:** Team TS3005's prototype costing in meeting d initially reached 18.50 Euros against the 12.50 Euro limit. Which major electronic feature did they drop, and what interface change did Speaker C propose to bring the total down to 11.50 Euros?
- **Manual Gold (`TS3005c#00034, TS3005d#00554, TS3005d#00627, TS3005d#00629, TS3005d#00654, TS3005d#00661`):** After planning an LCD touch screen and speech recognition in meeting c, Team TS3005 dropped the sample speaker/sensor (speech recognition, worth 4 points) during meeting d's costing. Speaker C then proposed dropping the scroll wheel and relying solely on the touch screen, lowering the total to 11.50 Euros.
- **Predicted:** To bring the total cost down, they dropped the sample speaker and Speaker C proposed replacing the wood finish with a normal special colour.
- **Verified Citations:**
  - `[TS3005d | TS3005d#00627]`: `"Yeah, but it's only a half. But I think the only option is to drop the uh sample speaker."`
  - `[TS3005d | TS3005d#00640]`: `"Yeah, but a wood we can make it brown, dark brown, not wood."`

### C11 (`TS3008`) — `ok=True`, `qwen/qwen3.8-27b`
- **Question:** Why did Team TS3008 argue against an LCD display in meeting c, and what unresolved budget problem did Speaker A identify regarding the docking station during meeting d's costing?
- **Manual Gold (`TS3008c#00214, TS3008c#00215, TS3008d#00405, TS3008d#00511`):** In meeting c, Speaker A and Speaker D rejected an LCD display as fragile and mismatched with their sturdy, throwable rubber-and-plastic design (and Speaker D noted TVs would need to send data back). In meeting d, Speaker A pointed out that the docking station was missing from the cost sheet and could not be built for the remaining 30 cents.
- **Predicted:** In meeting c, the team argued against an LCD display because it contradicted their philosophy of simplicity and the design was already considered 'overdone' or 'too much' with the existing docking station. In meeting d, Speaker A identified that the docking station was missing from the cost list and that it would be impossible to manufacture one for the previously assumed thirty cents.
- **Verified Citations:**
  - `[TS3008c | TS3008c#00271]`: `"I don't think I j uh, and really, I don't see how the the L_C_D_ display is gonna add anything, you know, on a design level."`
  - `[TS3008d | TS3008d#00511]`: `"the problem now is that the There's no such thing as a docking station in this list, but we can all imagine that it would be impossible to make a docking station for thirty cents."`

### C12 (`TS3012`) — `ok=False`, `llm_error:AllProvidersFailed`
- **Question:** Team TS3012 considered including an LCD screen, speech recognition, and a yellow rubber case in meeting c. In meeting d, what initial total production cost did Speaker A calculate, and which two items did Speaker A cut to hit the 12.50 Euro limit while keeping the rubber case?
- **Manual Gold (`TS3012c#00168, TS3012c#00177, TS3012d#00080, TS3012d#00090, TS3012d#00109`):** In meeting d, Speaker A (Project Manager) calculated an initial production cost of 17 Euros for the design discussed in meeting c. To reach the 12.50 Euro limit while keeping the spongy rubber case and LCD, Speaker A cut speech recognition (bringing it to 13 Euros) and the special colour (delivering a black/grey base with optional separate covers).
- **Predicted:** *(None — failed with `llm_error:AllProvidersFailed` after exhausting free-tier daily limits; no answer fabricated.)*

---

## 5. Human Sign-Off Block

- **Reviewer Name / Initials:** `_______________________`
- **Review Date:** `_______________________`
- **Final Human Total Score (`/ 24` across 12 questions, or `/ 20` across 10 live-answered):** `_______`
- **Notes on Re-Running `C08` & `C12`:** Once daily free-tier limits reset (or with a paid API key), run `python -m catalyst.pipeline eval --questions eval/eval_questions.json --max-steps 3 --k 8 --out eval/eval_report.json`; incremental checkpointing automatically preserves the 10 verified rows and runs only `C08` and `C12`.
