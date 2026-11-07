# Physiological and facial "tells" for deception and bluffing: evidence base for an honest webcam poker demo

Research date: 2026-11-07. Scope: what webcam-measurable signals change with deception, stress or bluffing; effect sizes and reliability; baseline protocols; defensible wording and ethics for a UK/EU hackathon demo.

## 1. Meta-analytic evidence on cues to deception, and what the polygraph literature says about arousal vs deception

### Takeaway
Behavioural cues to deception are faint (median |d| about 0.10 across 158 cues), people detect lies at about 54% (chance is 50%), and even the larger reported cue effects may be inflated by publication bias and low power. The NRC polygraph report says physiological measures pick up arousal, which has many causes, so they cannot be read as a direct deception signal. An honest demo should say it measures arousal or stress, and should not claim to detect lies.

### Cited Findings
- DePaulo et al. (2003) meta-analysis covered 116 studies and 158 cues, 102 of them nonverbal. The median cue effect was d = |.10|, which is small by conventional benchmarks — [Informa/Psychology, Crime & Law 2023 citing DePaulo 2003](https://informahealthcare.com/doi/full/10.1080/1068316X.2023.2292504); [PMC10884279](https://pmc.ncbi.nlm.nih.gov/articles/PMC10884279)
- In DePaulo et al. (2003), the nonverbal cues reaching |d| >= 0.20 included overall nervousness/tension (d about 0.27) and higher voice pitch (k = 12, d = 0.21) — [typeset.io, moderators of nonverbal indicators meta-analysis](https://typeset.io/pdf/moderators-of-nonverbal-indicators-of-deception-a-meta-3kfqtv8271.pdf); [PMC10884279](https://pmc.ncbi.nlm.nih.gov/articles/PMC10884279)
- DePaulo et al. summarised their result as "the looks and sounds of deceit are faint". Vrij et al. (2019) describe the cues found so far as "faint and unreliable" — [Denault et al. 2020, PMC7767987](https://pmc.ncbi.nlm.nih.gov/articles/PMC7767987)
- Gaze aversion: a meta-analysis found no evidence that liars avoid eye contact, even though gaze aversion is believed worldwide to be the main sign of lying — [PMC10884279](https://pmc.ncbi.nlm.nih.gov/articles/PMC10884279); Global Deception Research Team (2006) listed it as a folk belief, cited in [PMC7767987](https://pmc.ncbi.nlm.nih.gov/articles/PMC7767987). Mann et al. (2012) found liars made *longer* deliberate eye contact — [PMC7767987](https://pmc.ncbi.nlm.nih.gov/articles/PMC7767987)
- Luke (2019), "Lessons From Pinocchio: Cues to Deception May Be Highly Exaggerated" (Perspectives on Psychological Science, doi 10.1177/1745691619838258), argues that the large cue effects in the literature may be inflated. Statistical power in parts of the literature is so low that real effects cannot be told apart from false positives. Luke warns against using possibly illusory cues for practical credibility assessment — [APS journal page](https://www.psychologicalscience.org/journals/perspectives/1745691619838258/); [preprint](https://typeset.io/pdf/lessons-from-pinocchio-cues-to-deception-may-be-highly-2h6vzs5dhk.pdf)
- Luke's analysis found flexible coding, under-reporting of non-significant results and more significant results than power would predict. In Monte Carlo simulations, wholly useless cues could still produce the existing literature — [PMC7767987 summarising Luke 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7767987)
- Bond & DePaulo (2006), Personality and Social Psychology Review 10(3):214-234, covered 206 documents and 24,483 judges. The mean accuracy was about 54% (weighted 53.46%). Judges showed truth bias, rating 55.23% of accounts truthful. Experts did not beat laypeople (k = 20, d = -0.03) — [PubMed 16859438](https://pubmed.ncbi.nlm.nih.gov/16859438/); [full text PDF](https://assets.aclu.org/live/uploads/document/foia/2006-Personality-and-Social-Psychology-Review-Accuracy-of-Deception-Judgements.pdf)
- Mann et al. (2020): at a border crossing, attending to smugglers' nervousness pushed detection *below* chance — [PMC7767987](https://pmc.ncbi.nlm.nih.gov/articles/PMC7767987)
- NRC (2003), *The Polygraph and Lie Detection*, found the theoretical rationale for the polygraph "quite weak", especially for differential fear or arousal between relevant and comparison questions. Physiological signs of arousal vary widely between people. Deception does not mean the same thing psychologically to everyone, and arousal has many causes besides deception — [NAP ch.1/summary](https://www.nationalacademies.org/read/10420/chapter/2)
- NRC (2003): the measures are inherently ambiguous, so more investment would bring "only modest improvements". In the lab with naive examinees the polygraph does better than chance, but errors are not rare. Lab studies tend to overstate accuracy, and field screening accuracy is likely lower. The evidence base is "scanty and scientifically weak" — [NAP](https://www.nationalacademies.org/read/10420/chapter/2); [NAP ch.4](https://www.nationalacademies.org/read/10420/chapter/4)

### Inferences
- The scientific consensus supports "arousal is measurable, but deception is not directly measurable". This is the framing the demo should use.
- Any per-hand verdict like "BLUFF" would claim accuracy that even polygraph-grade sensors cannot reach. A webcam signal is noisier than a contact polygraph.
- Showing the audience the 54% human baseline is a good hook: "humans are near coin-flip; let's see what your body does instead".

### Gaps
- I did not verify DePaulo (2003) cue-level values for pupil dilation or blinking from the primary table. A pupil figure of d about 0.39 is often quoted, but I found no source confirming it in this session, so treat it as unverified.
- Hartwig & Bond's multi-cue accuracy work was not retrieved.

## 2. Heart rate / HRV during lying, bluffing and stress; rPPG/webcam stress detection accuracy

### Takeaway
Acute social stress reliably raises heart rate, with a large standardized effect (ES about 0.85-0.89 in TSST meta-analyses; one study went from about 70 to 96 bpm). In the Concealed Information Test, heart rate often *decelerates* to recognised items instead (d* = 0.89). So "HR goes up when lying" is an oversimplification. Webcam rPPG errors (about 5-10 bpm MAE for still subjects under ordinary lighting, worse with motion and darker skin) are as large as the effect sizes, so only coarse trends over tens of seconds against a personal baseline can be defended.

### Cited Findings
- TSST meta-analysis in youth: heart rate ES = 0.89 (p < 0.001), salivary cortisol ES = 0.47 — [Seddon et al. 2020, PMC7739033](https://pmc.ncbi.nlm.nih.gov/articles/PMC7739033)
- Traditional vs virtual TSST meta-analysis: pre-post HR ES = 0.85 (traditional) and 0.98 (VR) — [search summary of meta-analysis, HKU hub](https://hub.hku.hk/handle/10722/337699)
- Single-study raw example: mean HR rose from 70.5 bpm before the TSST to a peak mean of 96.5 bpm during the task. This is a secondary source and I could not check the original — [Wikipedia: Trier Social Stress Test](https://EN.wikipedia.com/wiki/Trier_Social_Stress_Test)
- CIT meta-analysis (Meijer et al. 2014, Psychophysiology), lab mock-crime and personal-item paradigms: d* = 1.55 for skin conductance, 1.11 for respiration line length, 0.89 for heart rate, 1.89 for P300. SCR effects were moderated by motivation, number of questions and inclusion of innocent participants — [PubMed 24916920](https://pubmed.ncbi.nlm.nih.gov/24916920)
- A separate CIT study (not fully attributed) found heart-rate *deceleration* in suspects but not witnesses, read as arousal inhibition — [scite author page](https://scite.ai/authors/nathalie-klein-selle-1gmX3M) (treat as supplementary)
- UBFC-Phys (Meziati Sabour et al. 2023, webcam video of a TSST-like protocol): binary stress recognition reached 85.48% accuracy using remote pulse-rate-variability features. Control vs test stress-level classification reached 69.73%, and the 3-class task 63.09% — [paper PDF](https://lead.ube.fr/wp-content/uploads/2025/04/Meziati_Sabour_et_al._2023.pdf)
- Later UBFC-Phys results: Ziaratnia et al. (WACV 2024) used CCT-LSTM on face video plus rPPG and got 83.2% accuracy / 83.4% F1 (7-fold) — [CVF open access](https://openaccess.thecvf.com/content/WACV2024/papers/Ziaratnia_Multimodal_Deep_Learning_for_Remote_Stress_Estimation_Using_CCT-LSTM_WACV_2024_paper.pdf). Fontes et al. (Sensors 2024) reported up to 95.83%, but I did not verify their validation protocol — [MDPI Sensors 24(4):1096](https://mdpi.com/1424-8220/24/4/1096)
- rPPG HR error under realistic conditions (MMPD mobile-video dataset, MAE in bpm). POS: stationary 9.70, talking 8.05, head rotation 7.50, walking 17.05. Lighting: LED-high 7.26, natural light 10.71. Skin tone: type 3 5.76, type 6 11.17. Supervised TS-CAN did best on stationary (5.34) and light skin, and generalised poorly — [MMPD, arXiv 2302.03840](https://ar5iv.labs.arxiv.org/html/2302.03840)
- In good conditions (BH-rPPG study), CHROM reached 1.04 bpm MAE under medium illumination — [rPPG lighting study search result](https://arxiv.org/pdf/2305.04161v2); UBFC-rPPG was recorded with a Logitech C920 at 640x480, 30 fps, with limited skin-tone and lighting variety — [rPPG-Toolbox / related](https://arxiv.org/pdf/2210.00716)
- In poker, Sung & Pentland (MIT Media Lab) put HR, skin conductance and voice sensors on players in real-money no-limit Hold'em. They report that SC peaks and HRV tracked high-stress moments and that simple classifiers identified stress/bluffing. This is an abstract only; sample size and accuracy were not verified — [MIT HD LiveNet poker](https://hd.media.mit.edu/livenet/poker.php)

### Inferences
- HR rises reliably with sustained social-evaluative stress (minutes, as in the TSST). A single bluffing decision lasts seconds, and its HR change is probably smaller and may go either way (orienting deceleration vs arousal acceleration).
- With rPPG MAE of about 5-10 bpm, a demo should average over 10-30 s windows, show a trend line relative to the player's own baseline, and display signal quality. It should never show "+3 bpm = bluff".
- Pulse-rate variability needs clean beat-to-beat timing. On a laptop webcam it is fragile, so it suits a "stretch goal / experimental" label.
- Latency: no source in this session gave HR response latency in seconds for a single deceptive act. The cardiac response to a discrete stimulus is commonly described as unfolding over several beats; this is general knowledge and not cited here.

### Gaps
- No pooled TSST bpm mean difference found (only standardized ES).
- No peer-reviewed study found that uses webcam rPPG specifically for *deception* (as opposed to stress) with reported accuracy.
- Individual variability in HR reactivity (SD in bpm) was not retrieved. The NRC qualitatively notes wide individual differences.

## 3. Blink rate, pupil, gaze, face touching, micro-expressions: evidence and webcam/MediaPipe measurability

### Takeaway
The most defensible webcam signal is blink rate. Leal & Vrij (2008) found suppression during lying followed by a rebound afterwards. Replication is partial (the suppression replicated, the rebound less so), and the pattern is tied to cognitive load, not lying as such. It is easy to measure with MediaPipe eye landmarks via eye aspect ratio. Pupil dilation is not reliable on RGB webcams. Gaze aversion is a myth. Micro-expressions lack scientific support.

### Cited Findings
- Leal & Vrij (2008), "Blinking during and after lying", J. Nonverbal Behavior 32:187-194: liars' blink rate dropped while lying (vs baseline) and rose after the lie. Truth tellers did not show the rebound — [Springer](https://link.springer.com/doi/10.1007/s10919-008-0051-0); [Frontiers 2013 summary](https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00736/full)
- Mechanism: blinks fall under cognitive demand and rebound when demand lifts — [PMC3151614](https://pmc.ncbi.nlm.nih.gov/articles/PMC3151614)
- Replication caveat: a later study (Perelman 2014, PeerJ, "Detecting deception via eyeblink frequency modulation") replicated the decrease during lying but not the post-lie increase. Its reviewer suggested the effect may come from the effort of fabricating, not emotional arousal — [PeerJ reviews](https://peerj.com/articles/260/reviews); [article](https://peerj.com/articles/260)
- Eye aspect ratio (Soukupova & Cech 2016, CVWW): each frame's eye landmarks reduce to one scalar (EAR) that stays roughly constant while the eye is open and drops toward zero in a blink. Detection uses a threshold (around 0.2 in the paper's figure) or an SVM over an EAR window — [CVWW 2016 paper](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf); [PyImageSearch explanation](https://pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/); adjusted EAR for strong blinks: [PMC9044337](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044337/)
- Pupil via webcam: models for estimating pupil size from standard webcams are scarce and limited by low resolution. EyeDentify (51 participants, laptop webcam vs Tobii ground truth) exists as a research dataset — [arXiv 2407.11204](https://arxiv.org/html/2407.11204v1). Ambient lighting strongly shifts baseline pupil size — [Frontiers Neurology 2024](https://frontiersin.org/articles/10.3389/fneur.2024.1363190/full). Browser tools that reach eye-tracker accuracy relied on *infrared* webcams — [PMC8489024](https://pmc.ncbi.nlm.nih.gov/articles/PMC8489024)
- Gaze aversion: no evidence it signals lying (see section 1) — [PMC10884279](https://pmc.ncbi.nlm.nih.gov/articles/PMC10884279). Vrij et al. (2015) reported that spontaneous saccade rate differed between liars and truth tellers, but the effects are too small to classify individuals — [PMC7767987](https://pmc.ncbi.nlm.nih.gov/articles/PMC7767987)
- Micro-expressions: the Ekman micro-expression hypothesis "has no scientific support" (citing Porter & ten Brinke 2008) — [PMC7767987](https://pmc.ncbi.nlm.nih.gov/articles/PMC7767987). Burgoon (2018, Frontiers in Psychology) cites a study where only 2% of 700 coded high-stakes expressions were micro-expressions; false negatives are common — [PMC6158306](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6158306/)
- Jordan et al. (2019, J. Investigative Psychology and Offender Profiling 16(3):222-235): 90 participants were randomised to Ekman's METT training, bogus training or no training. METT did not improve lie detection, and overall accuracy was slightly below chance — [PDF](https://web.williams.edu/Psychology/Faculty/Kassin/files/Jordan%20et%20al%20(2019)%20-%20METT.pdf); [Phys.org](https://m.phys.org/news/2019-09-flaws-tool.html)
- Nervousness/tension: d about 0.27 in DePaulo 2003, one of the larger nonverbal cues, but still small — [typeset.io](https://typeset.io/pdf/moderators-of-nonverbal-indicators-of-deception-a-meta-3kfqtv8271.pdf)

### Inferences
Measurability with a laptop webcam plus MediaPipe Face Mesh / Face Landmarker:

| Signal | Measurable? | Evidence for deception/stress | Demo recommendation |
|---|---|---|---|
| Heart rate (rPPG, forehead/cheek ROI, POS/CHROM) | Yes, with about 5-10 bpm MAE; needs stillness and even light | Stress: large ES. Deception: indirect, direction varies | **Primary signal**, smoothed, baseline-relative, with a quality indicator |
| Blink rate (EAR from eye landmarks) | Yes, robust at 30 fps | Suppression under cognitive load; partial replication | **Primary signal**: show blinks/min vs baseline, and the "post-decision rebound" as a fun visual |
| Head/face movement, stillness | Yes (landmark velocity) | Burgoon's "rigidity effect" is suggestive; weak evidence overall | Optional "fidget meter", clearly labelled playful |
| Face touching | Partly (MediaPipe Hands + face box) | Folk cue, weak support | Optional, labelled playful |
| Smile / AU-like expression | Yes (blendshapes) | Expressions are under voluntary control; smiles during bluffs were seen in one small study | Show it, but do not interpret as a tell |
| Pupil dilation | Not reliably on RGB webcam | Good evidence in lab eye-tracking | **Do not use**, or mark as not measured |
| Gaze aversion | Yes (iris landmarks) | Myth | Do not present as a tell; possibly use as a "myth-busting" slide |
| Micro-expressions | No (needs high fps; the construct itself is disputed) | Not supported | **Do not claim** |

### Gaps
- No EAR accuracy numbers (precision/recall) retrieved for MediaPipe specifically.
- Primary effect sizes for Leal & Vrij (2008) were not retrieved (sample was small; exact blink counts not verified).

## 4. Poker-specific research: tells and whether physiological arousal tracks bluffing

### Takeaway
In professional poker video, faces carried no usable information: judgements were at or slightly below chance, consistent with deliberate "poker faces". Arm motion while betting did predict hand quality (Slepian et al. 2013). Small physiological studies find that bluffing raises skin conductance and sometimes HR. One thesis found that bluffing and strong-hand value betting produced *similar* arousal, so arousal tracks high stakes/excitement, not deception specifically.

### Cited Findings
- Slepian, Young, Rutchick & Ambady (2013), Psychological Science (doi 10.1177/0956797613487384): 78 novices rated 2-s World Series of Poker clips. Judgements from faces were at or slightly below chance, upper body was about chance, and arms alone were the most accurate — [APS Observer](https://www.psychologicalscience.org/publications/observer/obsonline/to-call-a-players-poker-hand-look-to-the-arms-not-the-face.html); [APS "Calling a bluff"](https://www.psychologicalscience.org/news/were-only-human/calling-a-bluff-is-it-all-in-the-arms.html)
- Proposed mechanism: a separate group of 40 raters judged strong-hand players as more confident with smoother arm movements. Whether movements were actually smoother was not measured kinematically — [Science News](https://www.sciencenews.org/article/poker-pros-arms-betray-their-hands); [Scientific American podcast](https://www.scientificamerican.com/podcast/episode/poker-players-arms-give-away-their-13-04-22/)
- Lee (2013), University of Waterloo master's thesis "Deception and Arousal in Texas Hold 'em Poker": 71 players, 20 rounds. Bluffing produced skin-conductance responses *similar* to truthful strong bets (and to trapping), and wins were more arousing than losses. Unreviewed thesis — [UWSpace](https://uwspace.uwaterloo.ca/handle/10012/7500); [PDF](https://core.ac.uk/download/144146467.pdf)
- Sung & Pentland (MIT): wearable HR/SC/voice in real-money tournaments; stress moments were detectable; details unverified — [MIT](https://hd.media.mit.edu/livenet/poker.php)
- Nagasawa & Sumi (WCTP 2025 proceedings): 12 participants in a bluffing card game showed increased HR and EDA during bluffs, and smiles were frequent (possibly deliberate). Abstract only — [Atlantis Press](https://www.atlantis-press.com/proceedings/wctp-25/126023792)

### Inferences
- Faces are what players control best, so a webcam on the face will mainly capture involuntary autonomic signals (pulse, blinks). This fits the demo premise better than "reading expressions".
- Arousal also rises with strong hands and big pots. The honest output is "this moment was more arousing than your baseline", not "you are bluffing". A fun twist: reveal the cards after the hand, then show the arousal trace, and let the audience see that a strong hand and a bluff can look the same.

### Gaps
- No large, peer-reviewed, replicated study of physiological bluff detection in poker was found.
- No webcam-only poker bluff study was found.

## 5. Baseline-relative approaches and game protocols (CQT / CIT adaptations)

### Takeaway
Because physiological reactivity varies widely between people (NRC 2003), every signal must be expressed relative to the person's own resting baseline and to their own truthful/neutral responses. The Concealed Information Test (CIT) is the scientifically strongest paradigm (SCR d* = 1.55, HR d* = 0.89). It works by comparing responses to a known "relevant" item against equivalent foils within the same person. A "pick a card" game is a direct, honest adaptation: it detects *recognition/orienting*, not lying.

### Cited Findings
- NRC: physiological signs of arousal vary widely between people, and the CQT rationale (differential arousal to relevant vs comparison questions) is weak — [NAP](https://www.nationalacademies.org/read/10420/chapter/2)
- CIT meta-analysis effect sizes and moderators (motivation, number of questions, inclusion of innocents) — [PubMed 24916920](https://pubmed.ncbi.nlm.nih.gov/24916920)
- Leal & Vrij measured blink changes *relative to baseline* — [Frontiers 2013](https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00736/full)
- Lab-based accuracy overstates field accuracy — [NAP ch.4](https://www.nationalacademies.org/read/10420/chapter/4)

### Inferences
Recommended calibration and game protocol for the demo (design proposal, not tested):
1. **Signal check (10-15 s):** the player sits still, face lit evenly from the front; show the rPPG quality indicator and blink detection live. Only proceed if quality is good.
2. **Resting baseline (60 s):** neutral video or breathing; record median HR, HR spread and blink rate. Use rolling medians, not single frames.
3. **Truthful anchor (about 30-60 s):** the player answers easy true questions aloud (name, city), giving a "talking baseline", since talking itself changes blinks and rPPG quality.
4. **"Pick a card" CIT round:** the player secretly picks one of 5 cards. The app shows each card in random order, about 5-10 s each, with 2-3 repetitions, and the player says "no" to every card. The app ranks cards by the player's relative response (HR change in the window after onset, blink suppression). The output is "your body reacted most to card X". This is a recognition effect, and the app should say so. With 5 cards, chance is 20%, so show it as a guess with confidence, and expect frequent misses.
5. **Poker hands:** log per-decision windows (e.g. 10 s after seeing cards, during the bet). Show z-scores vs that player's baseline as an "arousal meter". Reveal cards at the end and plot arousal against hand strength and against bluff/no-bluff, so the audience sees the ambiguity themselves.
6. **Re-baseline** every few hands or after big laughs/movement; flag low-quality windows instead of guessing.

### Gaps
- No published webcam-based CIT accuracy found; the expected hit rate for a 5-card webcam CIT is unknown and should be measured on the team at the hackathon.

## 6. Ethics, consent, claims about individuals, and the EU AI Act

### Takeaway
The EU AI Act's Article 5(1)(f) has banned AI systems that infer emotions in workplaces and educational institutions since 2 February 2025 (except for medical/safety reasons). Emotion recognition elsewhere is classed high-risk. A voluntary hackathon entertainment demo is not obviously in the prohibited contexts. However, a hackathon at a university could be argued to sit near "education institutions", and the UK is not bound by the EU Act. The safest framing is to show raw physiological signals (pulse, blinks), not inferred emotions or honesty, with explicit opt-in consent and no storage.

### Cited Findings
- Article 5(1)(f) prohibits AI systems that infer emotions of natural persons in the workplace and education institutions, except for medical or safety reasons. The wording targets "AI systems to infer emotions", and the Commission guidelines extend it to inferring *intentions* — [FPF analysis](https://fpf.org/blog/red-lines-under-eu-ai-act-unpacking-the-prohibition-of-emotion-recognition-in-the-workplace-and-education-institutions/); [Sidley summary of Commission guidelines](https://www.sidley.com/en/-/media/resource-pages/ai-monitor/guidance/eu-commission-guidelines-on-prohibited-ai-practices-established-by-the-ai-act.pdf?la=en)
- Emotion recognition outside those settings is not prohibited but is "high risk" under Annex III. A call-centre example (tracking customers' emotions) falls outside the ban — [FPF](https://fpf.org/blog/red-lines-under-eu-ai-act-unpacking-the-prohibition-of-emotion-recognition-in-the-workplace-and-education-institutions/)
- The prohibitions have applied since 2 February 2025 — [Modulos](https://docs.modulos.ai/frameworks/eu-ai-act/prohibited-practices-and-transparency); [Freshfields](https://technologyquotient.freshfields.com/post/102kfql/eu-ai-act-unpacked-25-european-commission-releases-critical-ai-act-implementati)
- Commentators disagree on whether the guidelines limit 5(1)(f) to systems processing biometric data — [CMS](https://cms.law/en/mkd/legal-updates/eu-commission-issues-guidelines-on-prohibited-ai-practices-part-ii)
- Courtroom/forensic misuse of nonverbal "lie cues" is documented as a pseudoscience risk — [Denault et al. 2020, PMC7767987](https://pmc.ncbi.nlm.nih.gov/articles/PMC7767987)

### Inferences
- **Defensible claims:**
  - "Estimates your pulse from tiny colour changes in your skin (remote photoplethysmography)."
  - "Counts your blinks; blinking often drops when you are concentrating hard."
  - "Shows when your body looks more aroused or stressed than your own resting baseline."
  - "Arousal goes up with bluffs *and* with great hands, so it is a clue, not a verdict."
  - "Science says people spot lies about 54% of the time; this demo does not detect lies either."
- **Claims to avoid:** "lie detector", "bluff detector", "truth score", "detects deception/emotions", "reads micro-expressions", any per-person verdict about honesty, any accuracy percentage not measured by the team.
- **Suggested UI wording:** "Arousal meter", "Pulse (est.)", "Blinks/min vs your baseline", "Signal quality: good/poor", and a footer: "For fun. Not a lie detector. Physiological arousal has many causes."
- **Consent and data:** opt-in only, with a visible "camera on" indicator; process locally on the laptop/PYNQ board; store nothing by default (no video saved, no identities); delete session data on reset; offer a "spectator mode" with no measurement. Do not run it on passers-by. Do not use it on hackathon staff or judges as an assessment.
- **UK note:** UK GDPR still applies. Facial video used to derive physiological data could count as biometric / special-category data if used to identify people. Local processing with no storage minimises this. (Inference; UK-specific legal sources were not retrieved.)
- **Framing for judges:** position the project as "making the science of tells visible, and debunking lie-detector myths". This turns the honest limitation into the project's message.

### Gaps
- I did not retrieve the Official Journal text of Regulation (EU) 2024/1689 directly; the findings come from law-firm/FPF secondary sources.
- No UK ICO guidance on emotion AI was retrieved in this session (the ICO has published cautions on "emotion analysis" technologies, but this is unverified here).
