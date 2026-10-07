# Workflow patterns that keep LLM/VLM agents on generic, reliable steps inside a closed goal

Context for the reader: the EyeForge pipeline today sends one fixed prompt to Kimi-K3 (image in, JSON out: support surface, objects, fixed furniture), builds an Isaac Sim scene, renders a sheet, and Qwen scores it 0-10. The score never goes back into the plan. Notes checked on 2026-10-06; every link was opened during this research unless marked otherwise.

## 1. ReAct, Plan-and-Solve, Least-to-Most: references, core idea, gains

### Takeaway
All three break the task into explicit steps before (or while) answering. ReAct interleaves thought, action and observation. Plan-and-Solve writes a plan first and then runs it. Least-to-Most splits the problem into ordered subproblems and solves them one after another, feeding each answer into the next. All three report large gains over one-shot answers, but they were measured on text QA, maths and symbolic tasks, not on vision or 3D layout.

### Cited Findings
- **ReAct**: "ReAct: Synergizing Reasoning and Acting in Language Models", Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao. arXiv:2210.03629 (v1 2022-10-06), ICLR 2023. Interleaves reasoning traces with task actions (e.g. Wikipedia API calls), so reasoning updates the plan and actions bring in outside information. — [arXiv](https://arxiv.org/abs/2210.03629)
- ReAct reports +34% and +10% absolute success rate over imitation and RL baselines on ALFWorld and WebShop, prompted with only 1-2 in-context examples. On HotpotQA and FEVER it "overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning". — [arXiv](https://arxiv.org/abs/2210.03629)
- **Plan-and-Solve (PS / PS+)**: "Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models", Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Peng Lim. arXiv:2305.04091 (2023-05-06), ACL 2023. PS first devises "a plan to divide the entire task into smaller subtasks", then carries out the subtasks. PS+ adds more detailed instructions (e.g. extract the relevant variables, compute carefully) to cut calculation and missing-step errors. — [arXiv](https://arxiv.org/abs/2305.04091)
- PS+ "consistently outperforms Zero-shot-CoT across all datasets by a large margin" on 10 datasets, matches or beats Zero-shot Program-of-Thought, and is comparable to 8-shot CoT on maths. — [arXiv](https://arxiv.org/abs/2305.04091)
- **Least-to-Most**: "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models", Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, Ed Chi. arXiv:2205.10625 (2022-05-21), ICLR 2023. Two stages: break the problem into simpler subproblems, then solve them in order, each step using the answers already found. — [arXiv](https://arxiv.org/abs/2205.10625)
- On SCAN (compositional generalization) Least-to-Most reaches 99%+ on every split with 14 exemplars, against 16% for chain-of-thought. Specialized neural-symbolic models needed over 15,000 training examples to do as well. — [arXiv](https://arxiv.org/abs/2205.10625)

### Inferences
- For scene planning, Least-to-Most maps naturally onto the dependency order of the scene: (1) camera and horizon, (2) support surface and its plane, (3) fixed furniture and occluders, (4) objects of interest and their placement on (2) and (3), (5) lighting. Each step's JSON goes into the next prompt. Today's single call asks the model to settle all of this at once.
- Plan-and-Solve's "PS+" idea (spell out the error-prone steps in the instruction) is cheap to adopt. List the known failure modes (floating objects, wrong scale, wrong support plane) as explicit checks inside the prompt.
- ReAct only pays off when there are actions whose observations carry new information. Here, the useful actions would be "render a preview", "measure the plane fit" or "look up the asset catalog", with the observation fed back. Without such tools, ReAct turns into plain CoT.

### Gaps
- None of the three papers measure VLM spatial planning or 3D scene layout. Whether their gains carry over to Kimi-K3 or Qwen on photo-to-scene tasks is untested.

## 2. MetaGPT and SOPs encoded in prompts

### Takeaway
MetaGPT turns a human standard operating procedure into a fixed chain of roles. Roles pass each other structured documents, not free chat, and an executable check loops back up to 3 times. The ablation shows that executable feedback alone is worth about 4-5 points.

### Cited Findings
- "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework", Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, Jürgen Schmidhuber. arXiv:2308.00352 (v1 2023-08-01, last revised 2024-11-01). Oral at ICLR 2024. — [arXiv](https://arxiv.org/abs/2308.00352); [ICLR 2024 oral](https://iclr.cc/virtual/2024/oral/19756); [OpenReview](https://openreview.net/forum?id=VtmBAGCN7o)
- It "encodes Standardized Operating Procedures (SOPs) into prompt sequences", in an "assembly line paradigm" where each agent role handles one subtask. — [arXiv](https://arxiv.org/abs/2308.00352)
- **How the SOP is represented**: five roles (Product Manager, Architect, Project Manager, Engineer, QA Engineer), each defined by name, profile, goal and constraints. Roles hand off formal deliverables: PRDs, system designs with file lists, data structures and interface definitions, and diagrams. Messages go to a shared publish-subscribe pool that roles subscribe to, instead of agent-to-agent chat. — [arXiv HTML v7](https://arxiv.org/html/2308.00352v7)
- **Executable feedback**: the Engineer writes and runs unit tests, then debugs on failure, with at most 3 retries. — [arXiv HTML v7](https://arxiv.org/html/2308.00352v7)
- **Results**: HumanEval 85.9% and MBPP 87.7% Pass@1. On SoftwareDev: executability 3.75/4, 0.83 human corrections per task (ChatDev 2.5) and 124.3 tokens per code line (ChatDev 248.9). Removing executable feedback costs 4.2 points on HumanEval and 5.4 on MBPP. — [arXiv HTML v7](https://arxiv.org/html/2308.00352v7)

### Inferences
- The transferable idea is "fixed roles plus typed hand-off documents plus an executable check", not multi-agent chat. For EyeForge, the SOP could be: Observer (describes the photo: camera height, horizon, surfaces, lighting, clutter), then Planner (maps the observation to catalog assets and placements, as JSON), then Builder (deterministic code, Isaac), then Checker (renders and verifies). Each hand-off is a JSON document with a schema.
- MetaGPT's checks are executable (unit tests). The equivalent here is cheap programmatic checks before any VLM judge: objects resting on the support plane, no interpenetration, object pixel sizes in a sane range, the camera frustum containing the targets. The VLM score should be the last check, not the only one.

### Gaps
- MetaGPT is measured on software tasks only. No evidence found of SOP-style role chains being benchmarked on visual scene reconstruction.

## 3. Anthropic "Building effective agents" (Dec 2024)

### Takeaway
Anthropic separates workflows (LLMs and tools on predefined code paths) from agents (the LLM directs its own process). It advises starting with the simplest setup and adding agency only when that measurably helps. For a closed task with fixed subtasks and clear criteria, prompt chaining with programmatic gates plus an evaluator-optimizer loop is the recommended shape.

### Cited Findings
- "Building effective agents", Erik Schluntz and Barry Zhang, Anthropic, published 2024-12-19. — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- Definitions: workflows are "systems where LLMs and tools are orchestrated through predefined code paths". Agents are "systems where LLMs dynamically direct their own processes and tool usage". — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- Start simple: "For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough". Add complexity only when it demonstrably helps. — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- **Prompt chaining**: sequential calls, each working on the previous output, optionally with programmatic checks ("gates") between steps. Ideal "for situations where the task can be easily and cleanly decomposed into fixed subtasks", trading latency for accuracy. — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- **Routing**: classify the input, then send it to a specialized prompt. For "complex tasks where there are distinct categories that are better handled separately". — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- **Parallelization**: sectioning (independent subtasks) or voting (several attempts at the same task). — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- **Orchestrator-workers**: for tasks "where you can't predict the subtasks needed". — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- **Evaluator-optimizer**: one LLM generates, another evaluates and gives feedback, in a loop. Effective "when we have clear evaluation criteria, and when iterative refinement provides measurable value". — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- **Agent-computer interface (appendix)**: "Give the model enough tokens to 'think' before it writes itself into a corner". Keep formats close to text the model has seen. Avoid formatting overhead such as escaping. "Poka-yoke your tools" so arguments are hard to misuse. The authors spent more time optimizing tools than the overall prompt for SWE-bench, e.g. requiring absolute file paths removed a class of errors. — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

### Inferences
- The current EyeForge pipeline is half of an evaluator-optimizer: Qwen evaluates, but nothing is optimized. The smallest useful change is to send Qwen's structured critique (which object is wrong, why, and a suggested fix) back to the planner for 1-2 rounds. Keep the best-scoring plan, and stop when the score crosses a threshold.
- Routing fits the scene type: tabletop, floor, outdoor road, shelf. Each route gets its own SOP prompt and few-shot examples, instead of one generic prompt covering every scene.
- Voting (parallelization) is a cheap reliability lever for the JSON plan: sample N plans, then pick by programmatic checks plus judge score.
- "Poka-yoke" applies to the JSON schema itself. Use enums for asset categories and support types, metric units stated in field names (e.g. `height_m`), and placements relative to named surfaces rather than absolute world coordinates.

### Gaps
- The post gives qualitative guidance from customer work, not controlled measurements.

## 4. Structured outputs and "observe before deciding"

### Takeaway
Constrained decoding makes JSON valid by construction. Whether strict formats hurt reasoning is disputed: one paper found they do, and a rebuttal found that with matched prompts structured output equals or beats free text. The consistent practical answer is to let the model reason or describe first, inside or before the JSON (a free-text or `reasoning` field placed before the decision fields). For VLMs specifically, generating a scene graph first and then answering improves compositional reasoning.

### Cited Findings
- OpenAI Structured Outputs (launched 2024-08-06 with gpt-4o-2024-08-06): with `strict: true`, constrained decoding enforces the supplied JSON Schema. On OpenAI's complex-schema evals it scores 100%, against under 40% for gpt-4-0613. — [OpenAI](https://openai.com/index/introducing-structured-outputs-in-the-api/) (page returned 403 to the fetcher; figures confirmed through search snippets of that page and [Azure's announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-gpt-4o-2024-08-06-api-with-structured-outputs-on-azure/4232684))
- "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models", Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, Yun-Nung Chen. arXiv:2408.02442 (2024-08-05, revised 2024-10-14). Reports a "significant decline in LLMs reasoning abilities under format restrictions", worse the stricter the format, with reasoning tasks hit hardest. — [arXiv](https://arxiv.org/abs/2408.02442)
- The .txt (Outlines) rebuttal "Say What You Mean" says the paper compared structured and unstructured runs with different prompts, used a weak JSON prompt and an LLM parser. With matched prompts, structured generation came out equal or better: GSM8K 0.77 to 0.78, Last Letter 0.73 to 0.77, Shuffle Object 0.41 to 0.44. — [.txt blog](https://blog.dottxt.ai/say-what-you-mean.html); disputes [Tam et al.](https://arxiv.org/abs/2408.02442)
- Anthropic's agent guidance: give the model tokens to think before it commits, and keep formats close to natural text. — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
- **Describe-then-decide for VLMs**: "Compositional Chain-of-Thought Prompting for Large Multimodal Models", Chancharik Mitra, Brandon Huang, Trevor Darrell, Roei Herzig. arXiv:2311.17076, CVPR 2024, pp. 14420-14431. Zero-shot: "we first generate an SG [scene graph] using the LMM, and then use that SG in the prompt to produce a response". This improves compositional benchmarks and general multimodal benchmarks for several LMMs, with no fine-tuning. — [arXiv](https://arxiv.org/abs/2311.17076); [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Mitra_Compositional_Chain-of-Thought_Prompting_for_Large_Multimodal_Models_CVPR_2024_paper.html)

### Inferences
- Split Kimi-K3's single call into (a) an observation step that produces a structured scene graph or description of the photo (objects, which surface they sit on, relations, camera pose estimate, lighting), and (b) a planning step that maps that graph to the build JSON. Alternatively, keep one call but order the schema `observations -> reasoning -> plan`. With constrained decoding, fields are generated in schema order, so the decision fields come after the observation tokens.
- Use schema enforcement (Nebius/vLLM-style guided JSON, if available for Kimi-K3; not verified here) to remove parse failures, but keep a free-text field for reasoning so the dispute above stops mattering.
- The scene-graph intermediate also makes the judge's feedback addressable. Qwen can point to a node ("mug_2 floats above the table") and the planner can fix just that node.

### Gaps
- The CCoT abstract gives no per-benchmark numbers; the full paper was not opened.
- No primary source found that directly measures "reasoning field before answer fields" in JSON schemas. The ordering argument rests on autoregressive decoding plus Anthropic's guidance.
- Whether Nebius Token Factory supports strict JSON-schema decoding for Kimi-K3 was not checked.

## 5. Tool and capability manifests given to the model

### Takeaway
Telling the model exactly what it can use (tool documentation, or here the asset catalog and the builder's capabilities) is as effective as demonstrations, or more, and scales better. Toolformer shows tool use can also be learned, but for a closed pipeline, documentation in the prompt is the practical lever.

### Cited Findings
- "Tool Documentation Enables Zero-Shot Tool-Usage with Large Language Models", Cheng-Yu Hsieh, Si-An Chen, Chun-Liang Li, Yasuhisa Fujii, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, Tomas Pfister. arXiv:2308.00675 (2023-08-01). Zero-shot prompts with only tool documentation match few-shot prompts with demonstrations on existing benchmarks. On a new dataset with hundreds of tool APIs, documentation alone "significantly outperform[s]" few-shot without documentation. This holds across vision and language tasks, including image generation and video tracking. Documentation also avoids having to choose which demonstrations to include as the tool set grows. — [arXiv](https://arxiv.org/abs/2308.00675)
- "Toolformer: Language Models Can Teach Themselves to Use Tools", Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom. arXiv:2302.04761 (2023-02-09). The model learns, self-supervised from a handful of demonstrations per API, when to call tools (calculator, search, QA, etc.). It gets "substantially improved zero-shot performance", often competitive with much larger models, without losing language-modeling ability. — [arXiv](https://arxiv.org/abs/2302.04761)
- Anthropic ACI guidance: describe each tool as you would document it for a junior developer, and check: "Is it obvious how to use this tool, based on the description and parameters?" — [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

### Inferences
- Give the planner a "capability manifest": the list of available asset classes (with real-world size ranges), supported surface primitives, light types, and what the builder cannot do (no deformable objects, no transparent liquids, etc.). This should reduce plans that ask for impossible things, which the builder then has to drop silently.
- Per Hsieh et al., documentation scales better than adding more few-shot scene examples as the catalog grows. Keep at most 1-2 full examples, chosen per route.

### Gaps
- The conference venue for Hsieh et al. and Toolformer was not checked in this session (Toolformer is commonly cited as NeurIPS 2023; unverified here).

## 6. 2024-2026 work on reliability of agent workflows (flow engineering, DSPy, self-correction)

### Takeaway
A designed multi-stage flow with tests beats prompt tweaking by large margins (AlphaCodium: 19% to 44%). Prompt pipelines can be optimized automatically against a metric (DSPy). Self-refinement helps when the feedback comes from outside the model, but intrinsic self-correction often fails or makes reasoning worse. So the Qwen judge must be a different model with concrete criteria, ideally backed by programmatic checks.

### Cited Findings
- "Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering", Tal Ridnik, Dedy Kredo, Itamar Friedman. arXiv:2401.08500 (2024-01-16). A "test-based, multi-stage, code-oriented iterative flow" lifts GPT-4 on the CodeContests validation set from 19% to 44% pass@5, against a single well-designed direct prompt. — [arXiv](https://arxiv.org/abs/2401.08500)
- "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines", Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, Christopher Potts. arXiv:2310.03714 (2023-10-05). Pipelines are graphs of declarative modules with signatures (typed inputs and outputs). Optimizers ("teleprompters") tune instructions and demonstrations to maximize a metric. Within minutes of compiling: over 25% gain for GPT-3.5 and about 65% for Llama2-13b-chat over standard few-shot prompting. — [arXiv](https://arxiv.org/abs/2310.03714)
- "Self-Refine: Iterative Refinement with Self-Feedback", Aman Madaan, Niket Tandon, Prakhar Gupta et al. arXiv:2303.17651 (2023-03-30). The same LLM generates, critiques and refines, for about 20% absolute average improvement over single-pass output across 7 tasks (human and automatic metrics). — [arXiv](https://arxiv.org/abs/2303.17651)
- Limit: "Large Language Models Cannot Self-Correct Reasoning Yet", Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou. arXiv:2310.01798, ICLR 2024. "LLMs struggle to self-correct their responses without external feedback, and at times, their performance even degrades after self-correction." — [arXiv](https://arxiv.org/abs/2310.01798)
- MetaGPT's executable feedback (tests, at most 3 retries) adds 4-5 points (see section 2). — [arXiv HTML v7](https://arxiv.org/html/2308.00352v7)

### Inferences
- The EyeForge loop should be "flow engineered" like AlphaCodium. Use deterministic tests first (physics settle, plane contact, bbox sizes, visibility of target objects in the camera), then the Qwen judge, then a bounded replan (2-3 rounds, keep the best). Self-Refine-style gains are plausible because the feedback (render plus a different-model judge plus program checks) is external, which is the condition Huang et al. say is needed.
- DSPy-style optimization fits once a metric exists. EyeForge has a natural end metric (student detector F1 on real test photos), plus a cheaper proxy (the judge score). Tuning the planner prompt or its few-shot examples against the proxy, then checking against real F1, matches the project's "measure before switching" rule. Caveat: a judge-score proxy can be gamed; the HOPE test showed unreviewed synthetic data hurts, so the final arbiter has to stay real-photo F1.
- Keep the loop bounded and logged. Unbounded agent loops add latency and cost, and the guidance in section 3 favors fixed workflows when subtasks are known, which they are here.

### Gaps
- Venues for AlphaCodium, DSPy and Self-Refine were not confirmed in this session (DSPy is widely cited as ICLR 2024 and Self-Refine as NeurIPS 2023; unverified here).
- No 2025-2026 controlled study was found applying evaluator-optimizer loops to VLM-planned 3D scene reconstruction for synthetic data. Closest related work (e.g. LLM layout planners such as Holodeck or LayoutGPT) was out of scope and not checked.
- How well Qwen's 0-10 score correlates with downstream detector F1 is unknown. This has to be measured in-house before using the score as an optimization target.
