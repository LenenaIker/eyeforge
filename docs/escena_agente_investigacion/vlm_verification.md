# VLM verification of generated images and scenes: closed questions vs scalar scores

Context: EyeForge rebuilds an Isaac Sim scene from one photo and today asks Qwen for one 0-10 score (keep if >= 3). Known defects: floating objects, sideways sky, camera inside a wall, mixed masks, dull exposure. Notes below are sourced; inferences are marked as such.

## 1. Decomposition into atomic yes/no questions (TIFA, DSG, VQAScore, GenEval, T2I-CompBench, Gecko)

### Takeaway
The text-to-image evaluation literature converged on "question generation + question answering" (QG/A): an LLM turns the spec into atomic yes/no questions, a VQA model answers each one, score = fraction passed. This beats single holistic scores in human correlation, but only if questions are atomic, non-redundant and dependency-aware (DSG), and answers read from yes-probability are more reliable than generated text (VQAScore). Detector-based checks (GenEval) are used where VQA is weak (counts, positions).

### Cited Findings
- TIFA: Hu, Liu, Kasai, Wang, Ostendorf, Krishna, Smith, "TIFA: Accurate and Interpretable Text-to-Image Faithfulness Evaluation with Question Answering", ICCV 2023, arXiv 2303.11897. GPT-3 in-context learning extracts elements, classifies them into 12 categories (object, counting, color, spatial, shape...) and writes yes/no and multiple-choice questions; questions are kept only if UnifiedQA answers them consistently (F1 > 0.7). Benchmark: 4,000 prompts, 25,000 questions. ([arXiv abs](https://arxiv.org/abs/2303.11897), [HTML](https://arxiv.org/html/2303.11897))
- TIFA human correlation: TIFA (mPLUG-large VQA) Spearman 0.60, Kendall 0.47, vs CLIPScore 0.33 / 0.23 and SPICE caption metric 0.33 / 0.23. mPLUG-large recommended as default (84.5% accuracy against human VQA answers). Human inter-annotator alpha 0.67 on faithfulness, 0.88 on VQA answers. Weakest categories: shapes, counting, spatial relations (66-73% accuracy). ([HTML](https://arxiv.org/html/2303.11897))
- DSG: Cho, Hu, Garg, Anderson, Krishna, Baldridge, Bansal, Pont-Tuset, Wang, "Davidsonian Scene Graph: Improving Reliability in Fine-grained Evaluation for Text-to-Image Generation", ICLR 2024, arXiv 2310.18235. Names three QG/A failure modes: hallucinated or omitted questions, duplicated questions, and inconsistent VQA answers (e.g. says no object exists but then describes its color). ([arXiv](https://arxiv.org/abs/2310.18235))
- DSG pipeline: three LLM steps (1) extract atomic tuples (entities, attributes, relations, globals), (2) turn each into a yes/no question, (3) build parent-child dependencies. At answer time, if a parent is answered "no", its children are marked invalid/zero (if "Is there a motorcycle?" = no, "Is the motorcycle blue?" is skipped). Best VQA (PaLI) matched human answers 73.8% overall, Spearman 0.571; entity questions ~83%, global/style ~47%; text rendering 13.7-24.9%; counting ~67% and fails above 5. Released DSG-1k (1,060 prompts). ([HTML](https://arxiv.org/html/2310.18235))
- VQAScore: Lin, Pathak, Li, Li, Xia, Neubig, Zhang, Ramanan, "Evaluating Text-to-Visual Generation with Image-to-Text Generation", arXiv 2404.01291 (2024; ECCV 2024). Score = P("Yes" | image, "Does this figure show '{text}'? Please answer yes or no."), read from token likelihood, not generated text. ([arXiv](https://arxiv.org/abs/2404.01291))
- VQAScore numbers: Winoground group score 46.0 vs CLIPScore 7.8, GPT-4V-Eval 36.3, TIFA 11.3; DrawBench AUROC 85.3 vs CLIPScore 49.1; TIFA160 pairwise accuracy 71.2 vs 54.1. Authors argue QG/A methods (TIFA, VQ2, DSG) produce nonsensical questions on complex prompts, and report LLaVA-1.5 answers "Yes" to ~80% of questions that should be "No" (yes-bias). GenAI-Bench: 1,600 prompts, 15,000+ human ratings, Krippendorff alpha 0.72 (image). ([HTML](https://arxiv.org/html/2404.01291))
- GenEval: Ghosh, Hajishirzi, Schmidt, "GenEval: An Object-Focused Framework for Evaluating Text-to-Image Alignment", arXiv 2310.11513 (Oct 2023; NeurIPS 2023 Datasets and Benchmarks). Uses an object detector plus a color classifier to check object co-occurrence, position, count, color at instance level; reports strong human agreement. ([arXiv](https://arxiv.org/abs/2310.11513))
- T2I-CompBench: Huang, Duan, Sun, Xie, Li, Liu, arXiv 2307.06350 (NeurIPS 2023; extended T2I-CompBench++ 2025). Picks a different checker per category: BLIP-VQA for attribute binding, UniDet detection for spatial (incl. 3D-spatial) and numeracy, MLLMs (GPT-4V, ShareGPT4V) analyzed as judges. 8,000 prompts. ([arXiv](https://arxiv.org/abs/2307.06350))
- Gecko: Wiles et al., "Revisiting Text-to-Image Evaluation with Gecko: On Metrics, Prompts, and Human Ratings", ICLR 2025 (spotlight), arXiv 2404.16820. Improved QG/A metric with skill-tagged prompts; 100K+ human annotations over 4 rating templates; shows the human rating template itself changes model rankings; Gecko correlates better with humans than prior metrics, including on TIFA160. ([arXiv](https://arxiv.org/abs/2404.16820))

### Inferences
- The EyeForge spec is not a text prompt but a defect list, so questions can be hand-written once per stage (fixed checklist) instead of LLM-generated per image. That removes the main QG/A error source cited by VQAScore and DSG (bad generated questions).
- DSG-style dependencies map well: "Is there a sky?" -> "Is the sky at the top of the image?"; "Is there a floor/ground?" -> "Does every object touch a surface?".
- Reading P(yes) from logits (VQAScore) allows a per-question threshold calibrated on a few labelled renders, instead of one global 0-10 cutoff. Qwen-VL via an API that exposes logprobs would allow this; otherwise use repeated sampling.
- Counting, small text, style/material and fine spatial questions are weak for VQA; defects like "camera inside a wall" or "object floating" may be better checked geometrically in Isaac (raycasts, bounding boxes vs ground plane) and only confirmed visually, as GenEval and T2I-CompBench route spatial/count checks to detectors.

### Gaps
- No exact Gecko correlation numbers collected (abstract only).
- GenEval exact human agreement figure not retrieved.

## 2. Reliability of VLM-as-a-judge: scoring vs pairwise vs yes/no, biases, checklists

### Takeaway
VLM judges agree with humans much better in pairwise comparison than in absolute scoring; absolute scores cluster (high-score tendency) and are unstable. Decomposed binary checklists raise agreement and cut variance in LLM judges. Known biases: position bias, verbosity, yes-bias; chain-of-thought can hurt VLM judges. Open models lag closed ones as judges.

### Cited Findings
- MLLM-as-a-Judge: Chen, Chen, Zhang, Liu, Wang, Zhou, Zhang, Wan, Zhou, Sun, ICML 2024 (oral), arXiv 2402.04788. Three settings: scoring, pair comparison, batch ranking. MLLMs show "human-like discernment" in pair comparison but "significant divergence" in scoring and batch ranking. ([arXiv](https://arxiv.org/abs/2402.04788))
- Numbers (GPT-4V): scoring Pearson 0.490 (Gemini 0.304); pair comparison accuracy 0.636 with ties, 0.773 without; batch ranking normalized Levenshtein 0.361. 3-step CoT dropped GPT-4V scoring from 0.557 to 0.299. Gemini and LLaVA mostly give ~4 points and rarely 1-2 (high-score tendency). LLaVA repeated order "ABCD" in 88.2% of batch rankings (position bias). Lengthier text got +0.6 to 0.75 points (verbosity). Majority consistency over 6 runs: 67.5% pairwise, 41.8% batch ranking. ([HTML](https://arxiv.org/html/2402.04788))
- VIEScore: Ku, Jiang, Wei, Yue, Chen, ACL 2024, arXiv 2312.14867. MLLM gives 0-10 semantic-consistency and perceptual-quality scores with reasons; GPT-4o Spearman 0.4 vs humans (human-human 0.45); open-source MLLMs substantially underperform; weaker on image editing. ([arXiv](https://arxiv.org/abs/2312.14867))
- CheckEval: Lee, Kim, Kim, Cho, Kang, Kang, Kim, "CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists", EMNLP 2025, arXiv 2403.18771. Replaces Likert ratings with decomposed boolean questions; improves average agreement across 12 evaluator models by 0.45 and reduces score variance while keeping human correlation. ([arXiv](https://arxiv.org/abs/2403.18771))
- TICK: "TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation", NeurIPS 2024, arXiv 2410.03608. LLM-generated, instruction-specific YES/NO checklist; pass rate as score. Exact agreement with human preferences 46.4% -> 52.2% vs direct scoring; giving checklists to human raters raised their agreement 0.194 -> 0.256. ([arXiv](https://arxiv.org/pdf/2410.03608), [NeurIPS](https://neurips.cc/virtual/2024/106582))
- Yes-bias: LLaVA-1.5 says "Yes" to ~80% of questions whose answer should be "No" ([VQAScore HTML](https://arxiv.org/html/2404.01291)).
- Low-level quality (exposure, blur, noise): Q-Bench (Wu et al., arXiv 2309.14181, ICLR 2024) finds MLLM low-level perception skills "preliminary... unstable and relatively imprecise". ([arXiv](https://arxiv.org/abs/2309.14181))

### Inferences
- A single 0-10 score with threshold 3 sits in the weakest regime reported (absolute scoring, open model, clustering at mid/high scores). Switching to one yes/no question per known defect is the direction both CheckEval and TICK support.
- Put the "bad" answer on the "yes" side sparingly: given yes-bias, phrase questions so that "yes" means "pass" only if validated; better, test both polarities on a small labelled set and keep the phrasing with higher accuracy.
- Ask for the answer directly (no long CoT) per MLLM-as-a-Judge, or test both.
- "Dull exposure" is a low-level property where MLLMs are weak (Q-Bench); a histogram/luminance check is cheaper and more reliable than a VLM.

### Gaps
- No found study measuring yes/no checklists specifically for image judges with open models (Qwen-VL) vs a 0-10 score; would need our own small labelled set.
- Calibration (ECE) of VLM yes-probabilities on render defects not found.

## 3. VLM critics in 3D/graphics loops and render-vs-reference

### Takeaway
3D works use VLMs mostly as pairwise or criteria-based critics on multi-view renders (GPTEval3D, BlenderAlchemy) or as refiners of rendered scenes (SceneCraft). Reported limits: viewpoint and prompt sensitivity, hallucination, position bias, cost. Recent work prefers symbolic/geometric checks for physical plausibility and reserves VLMs for semantic and orientation feedback.

### Cited Findings
- GPTEval3D: Wu, Yang, Li, Zhang, Liu, Guibas, Lin, Wetzstein, "GPT-4V(ision) is a Human-Aligned Evaluator for Text-to-3D Generation", arXiv 2401.04092 (Jan 2024; CVPR 2024). Pairwise comparison of two assets, each shown as 1, 4 or 9 views plus normal maps; five criteria (text alignment, 3D plausibility, texture details, geometry details, texture-geometry coherency); ensemble over perturbations (view count, augmentation, prompt wording, render type) to get stable probabilities; Elo ranking. Mean Kendall tau 0.710 vs CLIP similarity 0.568, aesthetic score 0.671. Limits: API cost, hallucination, position bias, quadratic pair count; general-public raters only reached Cohen kappa 0.3 vs experts 0.53. ([arXiv](https://arxiv.org/abs/2401.04092), [HTML](https://arxiv.org/html/2401.04092))
- SceneCraft: Hu, Iscen, Jain, Kipf, Yue, Ross, Schmid, Fathi, "SceneCraft: An LLM Agent for Synthesizing 3D Scene as Blender Code", arXiv 2403.01248 (2024, ICML 2024). Builds a scene graph, converts relations into numerical layout constraints in Python, renders, and uses a VLM to analyze renders and iteratively refine; library learning of reusable functions; scenes up to 100 assets. ([arXiv](https://arxiv.org/abs/2403.01248))
- BlenderAlchemy: Huang, Yang, Guibas, "BlenderAlchemy: Editing 3D Graphics with Vision-Language Models", arXiv 2404.17672 (2024, ECCV 2024). Edit generator + visual state evaluator search over Blender edits; text goals grounded with "imagined" reference images from a generator; shown on simple but tedious material, geometry and lighting edits. ([arXiv](https://arxiv.org/abs/2404.17672))
- Eval3D: Duggal et al., CVPR 2025, arXiv 2504.18509. Uses foundation models as probes and measures their inconsistency across views (e.g. rendered normals vs 2D-predicted normals, DINOv2 semantic consistency per 3D point), localizes artifacts, aligns with human judgment. ([CVPR paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Duggal_Eval3D_Interpretable_and_Fine-grained_Evaluation_for_3D_Generation_CVPR_2025_paper.pdf), [project](https://eval3d.github.io/))
- SceneCritic: Sengupta, Ao, Cascante-Bonilla, arXiv 2604.13035 (Apr 2026). States VLM judges scoring rendered views are "sensitive to viewpoint, prompt phrasing, and hallucination"; a symbolic ontology-based evaluator aligns better with humans; text-only LLMs can beat VLMs on semantic layout; image VLM feedback most useful for semantic and orientation corrections. ([arXiv](https://arxiv.org/abs/2604.13035))
- Physical plausibility in scene generation is commonly measured geometrically (collision ratio, floating rate, out-of-bounds rate), with VLM judges (e.g. Gemini, GPT-4o) as complement in NaLA, MetaSpatial, ScenePilot. (Search snippet only, not verified in full text: [NaLA](https://arxiv.org/pdf/2606.29395), [MetaSpatial](https://arxiv.org/html/2503.18470), [ScenePilot](https://arxiv.org/pdf/2608.30307), [phi-Scene](https://arxiv.org/pdf/2606.21596).)

### Inferences
- Render vs reference photo maps naturally to the pairwise setting where VLMs are most reliable: show photo and render side by side and ask atomic questions ("Is the horizon at the same height in both?", "Is the sky above in both?"), swapping left/right order and keeping only consistent answers to cancel position bias (GPTEval3D ensembling idea).
- For single SAM 3D meshes shown turned around: multi-view grids (4 views, optionally normals) as in GPTEval3D; ask per view or on a grid, aggregate by majority.
- Floating objects and camera-in-wall are physical facts available in Isaac (contact/raycast); per SceneCritic and the geometric metrics, check them symbolically first and use the VLM only for what geometry cannot see (mixed masks, wrong sky orientation semantics).
- Cost: N questions x V views x K repeats calls; with small images and short yes/no outputs (or logprob reads), batching all questions in one prompt is cheaper but loses independence; DSG/TIFA answer each separately.

### Gaps
- No paper found that evaluates VLM checklists for photo-to-sim scene reconstruction specifically (render vs single source photo).
- No sourced numbers on majority-vote gains for open VLMs on yes/no visual questions beyond MLLM-as-a-Judge consistency figures.
- NaLA/ScenePilot/phi-Scene claims only from search snippets; full texts not read.
