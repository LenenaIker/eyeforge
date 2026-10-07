# Memory of failures, experience libraries and skill libraries for LLM/VLM agents

Context: EyeForge pipeline where a VLM plans Isaac Sim scenes from one real photo. Options under study: (a) automatic per-scene log (plan, defects, fixes, outcome), (b) curated catalog of known failures (defect, detection, fix) injected in prompts, (c) later retrieval of past successful plans by SigLIP image similarity as few-shot examples. User question: one .md per scene into a vector DB?

Note on sources: newer papers (2025-2026) were verified by fetching arXiv / HF pages in this session. Classic papers (Reflexion, Self-Refine, Voyager, AWM, Generative Agents, CLIN, Huang et al., KATE, Lost in the Middle, AgentPoison) are cited with their arXiv links and abstract-level numbers from prior knowledge; their pages were not re-fetched here, so numbers should be spot-checked before quoting them in public material.

## 1. Mechanisms and reported gains of experience / memory methods

### Takeaway
All successful methods store lessons as short natural-language or code items that are written only after an outcome signal (tests, environment success, judge), retrieved by task similarity, and kept small by editing, voting or deletion. Reported gains are large (10 to 50% relative) but almost always on benchmarks with a reliable success signal.

### Cited Findings
- Reflexion (Shinn et al., NeurIPS 2023): after a failed trial the agent writes a verbal self-reflection into an episodic buffer and retries; no weight updates. Reports 91% pass@1 on HumanEval vs 80% for GPT-4, and in the paper body +22% on AlfWorld and +20% on HotPotQA over baselines. The feedback signal is external (unit tests, environment reward). ([arXiv 2303.11366](https://arxiv.org/abs/2303.11366))
- Self-Refine (Madaan et al., NeurIPS 2023): same model generates, critiques and refines in a loop; about 20% absolute average improvement across 7 tasks. The paper itself notes weak gains on math reasoning because the model often fails to detect its own errors. ([arXiv 2303.17651](https://arxiv.org/abs/2303.17651))
- Voyager (Wang et al., 2023, TMLR 2024): Minecraft agent with an ever-growing skill library of executable code, indexed by an embedding of each skill's description and retrieved by similarity; a skill is added only after a GPT-4 "critic" self-verification step confirms the task succeeded. Reports 3.3x more unique items, 2.3x longer distances, tech-tree milestones up to 15.3x faster; ablations show removing self-verification causes the largest drop. ([arXiv 2305.16291](https://arxiv.org/abs/2305.16291))
- ExpeL (Zhao et al., AAAI 2024): two parts. (1) Insight extraction: an LLM compares successful and failed trajectories from training tasks and maintains a list of natural-language insights via operations ADD, EDIT, UPVOTE, DOWNVOTE (insights with low votes are removed). (2) At inference, retrieves the most similar successful trajectories as few-shot examples plus the insight list. Performance grows as experience accumulates. ([arXiv 2308.10144](https://arxiv.org/abs/2308.10144))
- Agent Workflow Memory (Wang et al., 2024, ICML 2025): induces reusable "workflows" (abstracted sub-routines) from past successful trajectories and adds them to memory; +24.6% and +51.1% relative success on Mind2Web and WebArena, with fewer steps. Works offline (from training examples) and online (from self-judged test-time experiences). ([arXiv 2409.07429](https://arxiv.org/abs/2409.07429))
- Generative Agents (Park et al., UIST 2023): memory stream of observations; retrieval score = recency + importance + relevance (embedding similarity); periodic "reflections" synthesize higher-level memories. Ablation shows memory, reflection and planning each contribute to believability. ([arXiv 2304.03442](https://arxiv.org/abs/2304.03442))
- CLIN (Majumder et al., 2023): continually updated memory of causal abstractions ("X is necessary for Y", "X does not contribute to Y") rewritten after each trial; on ScienceWorld it reports outperforming Reflexion by 23 absolute points and transfer to new environments/tasks. ([arXiv 2310.10134](https://arxiv.org/abs/2310.10134))
- Dynamic Cheatsheet (Suzgun, Yuksekgonul, Bianchi, Jurafsky, Zou; arXiv 2025, EACL 2026): a black-box model keeps a self-curated, persistent "cheatsheet" of concise transferable strategies and code snippets across queries. Claude 3.5 Sonnet more than doubled accuracy on AIME; GPT-4o on Game of 24 went from about 10% to 99% after discovering a reusable Python solution. ([arXiv 2504.07952](https://huggingface.co/papers/2504.07952), [ACL Anthology](https://aclanthology.org/2026.eacl-long.333/), [code](https://github.com/suzgunmirac/dynamic-cheatsheet))
- Agent KB (Tang et al., 2025): hierarchical shared experience base (high-level strategies plus detailed execution logs) with a Reason-Retrieve-Refine pipeline, usable across agent frameworks. GAIA success up to +16.28 points (e.g. 38.46% to 57.69% on hardest tasks for one model); SWE-bench 41.33% to 53.33%. ([arXiv 2507.06229](https://arxiv.org/abs/2507.06229), [code](https://github.com/OPPO-PersonalAI/Agent-KB))
- ReasoningBank (Ouyang et al., Google, ICLR 2026): distills generalizable reasoning strategies from both self-judged successful and failed trajectories; each memory item has title, description, content; retrieved by similarity at test time and new learnings written back. Up to +8.3 success points on WebArena, 16% fewer interaction steps, up to 34.2% relative improvement with memory-aware test-time scaling (MaTTS). Beats memories that store raw trajectories or only successful routines. Acknowledged limitation: relies on LLM-as-a-judge for correctness without ground truth. ([arXiv 2509.25140](https://arxiv.org/abs/2509.25140), [HF summary](https://huggingface.co/papers/2509.25140))
- ACE, Agentic Context Engineering (Zhang et al., Stanford/SambaNova/Berkeley, 2025): treats the prompt context as an evolving "playbook" of bullets with ids and helpful/harmful counters, maintained by Generator, Reflector and Curator roles via incremental delta updates and "grow-and-refine" with semantic dedup. +10.6% on agent benchmarks, +8.6% on finance. ([arXiv 2510.04618](https://arxiv.org/abs/2510.04618))
- Survey: Zhang et al. 2024 categorizes agent memory by source, form (textual vs parametric) and operations (writing, management, reading). ([arXiv 2404.13501](https://arxiv.org/abs/2404.13501))

### Inferences
- Two families map directly onto the EyeForge options: "insight lists / cheatsheets / playbooks" (ExpeL insights, Dynamic Cheatsheet, ACE, CLIN) correspond to the curated failure catalog; "retrieve similar past successes" (ExpeL trajectories, Voyager skills, AWM workflows, ReasoningBank) corresponds to SigLIP retrieval of past plans.
- ReasoningBank and ExpeL both show that lessons distilled from failures plus successes beat storing raw trajectories, which argues against dumping full per-scene .md logs as the retrievable unit.

### Gaps
- No found work applies these memory methods to VLM planning of 3D simulation scenes or synthetic-data generation specifically; transfer to our domain is an inference.

## 2. Risks: wrong lessons, memory pollution, self-correction without external feedback

### Takeaway
The evidence is consistent: memory helps only when the write step is gated by a reliable outcome signal. Self-judged lessons without external feedback can degrade performance, bad memories propagate via "experience-following", and abstracted playbooks can collapse when rewritten wholesale.

### Cited Findings
- Huang et al. (ICLR 2024), "Large Language Models Cannot Self-Correct Reasoning Yet": intrinsic self-correction (no oracle labels) does not improve and often degrades reasoning accuracy; earlier reported gains relied on oracle labels to decide when to stop. ([arXiv 2310.01798](https://arxiv.org/abs/2310.01798))
- Kamoi et al. (TACL 2024) survey: self-correction works well with reliable external feedback (code execution, tools) but there is little evidence it works with feedback from the same prompted LLM on general tasks. ([arXiv 2406.01297](https://arxiv.org/abs/2406.01297))
- Xiong et al. 2025 (Xiong, Lin, Xie, He, Liu, Tang, Lakkaraju, Xiang): agents show "experience-following" (similar input to a retrieved record gives similar output). Consequences: error propagation (wrong stored executions are replicated and re-stored) and misaligned experience replay (some past successes mislead). Selective addition plus history-based deletion gave about 10% absolute average gain over naive memory growth; future task outcomes can be used as free labels to retroactively score stored records. ([arXiv 2505.16067](https://arxiv.org/abs/2505.16067), [overview](https://www.alphaxiv.org/overview/2505.16067v1))
- ACE documents "context collapse": on AppWorld a monolithically rewritten context went from 18,282 tokens (66.7% acc.) to 122 tokens (57.1%) in one step, below the 63.7% no-adaptation baseline. Also: on finance tasks without ground truth or reliable signals both ACE and Dynamic Cheatsheet degraded. ([arXiv 2510.04618 html](https://arxiv.org/html/2510.04618v1))
- ReasoningBank flags LLM-as-judge noise as a limitation for ambiguous tasks. ([HF](https://huggingface.co/papers/2509.25140))
- Voyager's design gates skill writes behind self-verification; Reflexion's gains depend on external signals (unit tests, env rewards). ([Voyager](https://arxiv.org/abs/2305.16291), [Reflexion](https://arxiv.org/abs/2303.11366))
- Security: AgentPoison (Chen et al., NeurIPS 2024) poisons agent memory / RAG knowledge bases with trigger-optimized records, reporting high attack success with a very small poison rate. MINJA (Dong et al.) injects malicious records using only queries, through the agent's own experience-extraction step: 98.2% injection success, 76.8% attack success. ([AgentPoison arXiv 2407.12784](https://arxiv.org/abs/2407.12784), [MINJA arXiv 2503.03704](https://arxiv.org/abs/2503.03704))

### Inferences
- For EyeForge, the VLM reviewer (Qwen score >= 3) is itself an LLM judge; lessons auto-written from its verdicts inherit its blind spots. The strongest available external signals are: human eye review, objective checks (geometry: floating/intersecting objects, empty renders, camera inside mesh), and ultimately the downstream student metric on real test (as in the HOPE test, where unreviewed synth hurt F1 0.814 vs 0.830 reviewed). These should gate promotion of lessons.
- Experience-following implies that retrieving a past plan for a visually similar photo will make the VLM copy it, including its defects. Only store plans as retrievable examples after they pass QC and, ideally, after their data helped the student.
- Security risk is low for a single-user internal pipeline, but if users' photos/specs drive automatic lesson writing in a multi-user product, MINJA-style pollution becomes relevant: keep user-triggered logs separate from the curated catalog.

### Gaps
- Exact Xiong et al. evaluator setup (ground truth vs LLM judge for selective addition) could not be confirmed from the fetched text.
- AgentPoison exact numbers not re-verified in this session.

## 3. Retrieval of in-context examples by similarity; vector DB vs whole catalog in the prompt

### Takeaway
Similarity-based retrieval of few-shot examples beats random when only a few examples fit (text and vision), but its advantage vanishes with many demonstrations in long context. A catalog of tens of short failure entries fits easily in context; a vector DB only pays off for a large, growing pool of examples (hundreds+) or image-keyed retrieval.

### Cited Findings
- Liu et al. (KATE, DeeLIO 2022): choosing in-context examples by kNN in sentence-embedding space consistently beats random selection on several NLU/generation tasks. ([arXiv 2101.06804](https://arxiv.org/abs/2101.06804))
- Zhang et al. (NeurIPS 2023), visual in-context learning: prompt retrieval clearly beats random; good examples are semantically similar and close in viewpoint, background and appearance; unsupervised retrieval uses off-the-shelf features with cosine distance, supervised retrieval works best. ([arXiv 2301.13670](https://arxiv.org/abs/2301.13670))
- Caveat from 2026 work: large multimodal models do not necessarily exploit visual similarity directly; retrieving counterfactual examples can help more than nearest neighbours. ([arXiv 2603.16737](https://arxiv.org/abs/2603.16737))
- Bertsch et al. (NAACL 2025): scaling from 10 to 1000 demonstrations gave up to +50.8 points; retrieval (BM25 etc.) beats random at few demonstrations, but at large counts they become indistinguishable, so a fixed random set (cacheable) is acceptable. ([arXiv 2405.00200](https://arxiv.org/abs/2405.00200), [ACL Anthology](https://aclanthology.org/2025.naacl-long.605/))
- Liu et al. "Lost in the Middle" (TACL 2024): models use information at the start/end of long contexts better than in the middle, so long unordered context has a cost. ([arXiv 2307.03172](https://arxiv.org/abs/2307.03172))
- ACE argues for keeping a detailed, itemized playbook in context rather than compressed summaries ("brevity bias"), with dedup to control size. ([arXiv 2510.04618](https://arxiv.org/abs/2510.04618))

### Inferences
- Failure catalog (defect, detection, fix): put the whole thing in the prompt while it is under roughly a few thousand tokens (order of 20 to 50 entries of 2 to 4 lines). No vector DB needed; ordering most-frequent/severe first mitigates lost-in-the-middle.
- Past plan retrieval: plans are long (JSON scene plans), so only 1 to 3 fit usefully; here similarity matters (few-shot regime per KATE/visual ICL). SigLIP embeddings plus a plain numpy cosine search over a .npy file is enough for hundreds to a few thousand scenes; a vector DB adds infra without benefit at that scale.
- The retrievable unit should be the structured plan plus outcome fields, not a free-form .md narrative; .md logs are fine for humans but embedding prose dilutes retrieval and invites copying narrative errors.

### Gaps
- No direct evidence found on whether VLM scene planners benefit from image-similar exemplars vs a fixed set of diverse exemplars; needs an A/B on our own scenes.

## 4. Practical recommendations: thresholds, human-in-the-loop promotion

### Takeaway
Literature supports a three-layer design: raw log (write everything, never in prompt), curated lessons (small, gated by verified outcome, edited not appended), retrieved exemplars (only verified successes, few). Promotion should be gated by external signals, with counters and deletion.

### Cited Findings
- ExpeL's insight list uses explicit ADD/EDIT/UPVOTE/DOWNVOTE so weak insights are pruned. ([arXiv 2308.10144](https://arxiv.org/abs/2308.10144))
- ACE stores per-bullet helpful/harmful counters and merges deltas instead of rewriting, avoiding collapse. ([arXiv 2510.04618](https://arxiv.org/abs/2510.04618))
- Xiong et al.: selective addition plus history-based deletion (drop records tied to repeated poor outcomes) is the most effective management, about +10% absolute. ([arXiv 2505.16067](https://arxiv.org/abs/2505.16067))
- Voyager: write only after verification. ([arXiv 2305.16291](https://arxiv.org/abs/2305.16291))
- ReasoningBank: lessons from failures are valuable, stored as short title/description/content items rather than raw trajectories. ([arXiv 2509.25140](https://arxiv.org/abs/2509.25140))
- Huang et al. / Kamoi et al.: do not trust self-generated critiques without external feedback. ([arXiv 2310.01798](https://arxiv.org/abs/2310.01798), [arXiv 2406.01297](https://arxiv.org/abs/2406.01297))

### Inferences (proposed design for EyeForge, not from literature directly)
- Layer 1, per-scene log (automatic): JSON per scene with photo id, SigLIP embedding id, plan, defects detected (by which checker), fixes applied, reviewer score, human verdict if any, downstream metric if any. Never injected wholesale into prompts.
- Layer 2, failure catalog (curated): candidate lessons auto-proposed by a reflector from logs, but promoted only when (a) the defect is confirmed by a human or an objective check and (b) the fix was observed to work at least on 2 to 3 scenes. Each entry: id, defect, how to detect, fix, evidence scene ids, helpful/harmful counts. Delete entries whose fix later fails repeatedly. Keep in full in the prompt while small.
- Layer 3, exemplar retrieval: index only scenes that passed QC (and ideally whose data helped the student). Retrieve top 1 to 3 by SigLIP cosine; start with numpy, move to a vector DB only if the pool reaches tens of thousands. A/B against fixed diverse exemplars before adopting, given experience-following risk.
- Answer to "a .md per scene into a vector DB": not recommended as the first step. Keep structured logs, curate a small catalog by hand-in-the-loop promotion, and add image-similarity retrieval later on verified plans only.

### Gaps
- No published numeric threshold (e.g. "N entries before RAG") was found; the size suggestions above are inferences from context-length findings, not measured guidance.
- No study found measuring human-in-the-loop promotion of agent lessons vs fully automatic curation.
