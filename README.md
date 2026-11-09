# Awesome RAG Papers

> **v2.0 draft:** Hierarchical organization preserving all 15 original research categories. Generated from a reconstructed prior inventory, **not freshly synchronized with GitHub main**.

**Inventory:** 148 original paper occurrences → 140 unique paper records. All occurrences remain mapped in [`data/alignment.csv`](data/alignment.csv).

For an overview of RAG and a guide to the 15-category taxonomy, see **[Introduction.md](Introduction.md)**.

## Contents

- [Foundations & Architecture](#foundations--architecture)
  - [Naive RAG](#naive-rag)
  - [Advanced Architectures](#advanced-architectures)
- [Retrieval & Context Engineering](#retrieval--context-engineering)
  - [Chunking/Indexing](#chunkingindexing)
  - [Query Enhancement](#query-enhancement)
  - [Context Optimization](#context-optimization)
  - [Long Context](#long-context)
- [Reasoning & Agentic RAG](#reasoning--agentic-rag)
  - [Multi-Hop & Knowledge Graphs](#multi-hop--knowledge-graphs)
  - [Iterative Retrieval & Reasoning](#iterative-retrieval--reasoning)
  - [Adaptive Retrieval](#adaptive-retrieval)
  - [Tools & Planning](#tools--planning)
- [Training & Trustworthiness](#training--trustworthiness)
  - [Preference Alignment](#preference-alignment)
  - [Knowledge Conflicts & Factuality](#knowledge-conflicts--factuality)
  - [Security & Privacy](#security--privacy)
- [Applications & Evaluation](#applications--evaluation)
  - [Specialized Domains & Modalities](#specialized-domains--modalities)
  - [Evaluation & Analysis](#evaluation--analysis)

---

## Foundations & Architecture

### Naive RAG

- [Benchmarking Retrieval-Augmented Generation for Medicine](https://aclanthology.org/2024.findings-acl.372.pdf)
- [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)
- [Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts](https://aclanthology.org/2024.findings-acl.458.pdf)
- [M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions](https://arxiv.org/pdf/2405.16420)
- [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/pdf/2401.00396)
- [Synergistic Interplay between Search and Large Language Models for Information Retrieval](https://arxiv.org/pdf/2305.07402)
- [To Generate or to Retrieve? On the Effectiveness of Artificial Contexts for Medical Open-Domain Question Answering](https://arxiv.org/pdf/2403.01924)
- [UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems](https://arxiv.org/pdf/2401.13256)

### Advanced Architectures

- [ClueAnchor: Clue-Anchored Knowledge Reasoning Exploration and Optimization for Retrieval-Augmented Generation](https://arxiv.org/pdf/2505.24388)
- [ComposeRAG: A Modular and Composable RAG for Corpus-Grounded Multi-Hop Question Answering](https://arxiv.org/pdf/2506.00232)
- [From RAG to RICHES: Retrieval Interlaced with Sequence Generation](https://arxiv.org/pdf/2407.00361)
- [GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](https://arxiv.org/pdf/2505.18710)
- [MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](https://arxiv.org/pdf/2501.00332)
- [OpenRAG: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models](https://arxiv.org/pdf/2410.01782)
- [Parametric Retrieval Augmented Generation](https://arxiv.org/pdf/2501.15915)
- [RAGLAB: A Modular and Research-Oriented Unified Framework for Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.11381)
- [RAGraph: A General Retrieval-Augmented Graph Learning Framework](https://arxiv.org/pdf/2410.23855)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/pdf/2310.11511)
- [UltraRAG: A Modular and Automated Toolkit for Adaptive Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.08761)

## Retrieval & Context Engineering

### Chunking/Indexing

- [Bottleneck-Minimal Indexing for Generative Document Retrieval](https://arxiv.org/pdf/2405.10974)
- [Chunk-Distilled Language Modeling](https://arxiv.org/pdf/2501.00343)
- [Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](https://arxiv.org/pdf/2410.11001)
- [Grounding Language Model with Chunking-Free In-Context Retrieval](https://arxiv.org/pdf/2402.09760)
- [Landmark Embedding: A Chunking-Free Embedding Method For Retrieval Augmented Long-Context Large Language Models](https://arxiv.org/pdf/2402.11573)
- [Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation](https://arxiv.org/pdf/2406.00456)
- [MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System](https://arxiv.org/pdf/2503.09600)

### Query Enhancement

- [ARL2: Aligning Retrievers with Black-box Large Language Models via Self-guided Adaptive Relevance Labeling](https://arxiv.org/pdf/2402.13542)
- [BlendFilter: Advancing Retrieval-Augmented Large Language Models via Query Generation Blending and Knowledge Filtering](https://aclanthology.org/2024.emnlp-main.58.pdf)
- [Diversify-verify-adapt: Efficient and Robust Retrieval-Augmented Ambiguous Question Answering](https://arxiv.org/pdf/2409.02361)
- [ExpandR: Teaching Dense Retrievers Beyond Queries with LLM Guidance](https://aclanthology.org/2025.emnlp-main.963.pdf)
- [MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2408.17072v1)
- [Optimizing Query Generation for Enhanced Document Retrieval in RAG](https://arxiv.org/pdf/2407.12325)
- [RaFe: Ranking Feedback Improves Query Rewriting for RAG](https://aclanthology.org/2024.findings-emnlp.49.pdf)
- [UniRAG: Unified Query Understanding Method for Retrieval Augmented Generation](https://openreview.net/attachment?id=h68SaHDtal&name=pdf)

### Context Optimization

- [Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection](https://arxiv.org/pdf/2405.16178)
- [Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Handling Noisy Contexts](https://aclanthology.org/2024.findings-emnlp.136.pdf)
- [An Information Bottleneck Perspective for Effective Noise Filtering on Retrieval-Augmented Generation](https://arxiv.org/pdf/2406.01549)
- [ECoRAG: Evidentiality-guided Compression for Long Context RAG](https://arxiv.org/pdf/2506.05167)
- [EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.12559)
- [FIT-RAG: Black-Box RAG with Factual Information and Token Reduction](https://arxiv.org/pdf/2403.14374)
- [Making Retrieval-Augmented Language Models Robust to Irrelevant Context](https://arxiv.org/pdf/2310.01558)
- [PISCO: Pretty Simple Compression for Retrieval-Augmented Generation](https://arxiv.org/pdf/2501.16075)
- [Provence: efficient and robust context pruning for retrieval-augmented generation](https://arxiv.org/pdf/2501.16214)
- [Small Models, Big Insights: Leveraging Slim Proxy Models To Decide When and What to Retrieve for LLMs](https://arxiv.org/pdf/2402.12052)
- [xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token](https://arxiv.org/pdf/2405.13792)

### Long Context

- [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.17031)
- [BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack](https://arxiv.org/pdf/2406.10149)
- [Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](https://arxiv.org/pdf/2505.10413)
- [Inference Scaling for Long-Context Retrieval Augmented Generation](https://arxiv.org/pdf/2410.04343)
- [Long-Context Inference with Retrieval-Augmented Speculative Decoding](https://arxiv.org/pdf/2502.20330)
- [Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG](https://arxiv.org/pdf/2410.05983)
- [LongRAG: A Dual-perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering](https://aclanthology.org/2024.emnlp-main.1259.pdf)
- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/pdf/2406.15319)
- [On the Role of Long-tail Knowledge in Retrieval Augmented Large Language Models](https://arxiv.org/pdf/2406.16367)

## Reasoning & Agentic RAG

### Multi-Hop & Knowledge Graphs

- [EfficientRAG: Efficient Retriever for Multi-Hop Question Answering](https://arxiv.org/pdf/2408.04259)
- [FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs](https://arxiv.org/pdf/2501.09957)
- [Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering](https://arxiv.org/pdf/2406.14891)
- [GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model Reasoning on Knowledge Graphs](https://arxiv.org/pdf/2405.20139)
- [Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG and Symbolic Verification](https://arxiv.org/pdf/2504.04578)
- [Hierarchical Retrieval-Augmented Generation Model with Rethink for Multi-hop Question Answering](https://arxiv.org/abs/2408.11875)
- [HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/pdf/2503.10150)
- [HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval Augmented Generation](https://arxiv.org/pdf/2502.12442)
- [RASPberry: Retrieval-Augmented Monte Carlo Tree Self-Play with Reasoning Consistency for Multi-Hop Question Answering](https://aclanthology.org/2025.findings-acl.587.pdf)
- [REANO: Optimising Retrieval-Augmented Reader Models through Knowledge Graph Generation](https://aclanthology.org/2024.acl-long.115.pdf)
- [Retrieve, Summarize, Plan: Advancing Multi-hop Question Answering with an Iterative Approach](https://arxiv.org/pdf/2407.13101)
- [SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.15272)

### Iterative Retrieval & Reasoning

- [DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models](https://arxiv.org/pdf/2403.10081)
- [KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.18397)
- [RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback](https://aclanthology.org/2024.findings-acl.281.pdf)
- [ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation](https://arxiv.org/pdf/2503.21729v1)
- [Retrieval-Augmented Generation by Evidence Retroactivity in LLMs](https://arxiv.org/pdf/2501.05475)
- [Retrieving, Rethinking and Revising: The Chain-of-Verification Can Improve Retrieval Augmented Generation](https://aclanthology.org/2024.findings-emnlp.607.pdf)
- [SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](https://arxiv.org/pdf/2410.13553v2)
- [Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience](https://arxiv.org/pdf/2506.00842)
- [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.05312)

### Adaptive Retrieval

- [Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](https://arxiv.org/pdf/2505.12731)
- [ACTIVERAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/pdf/2402.13547)
- [DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.10198)
- [SEAKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](https://arxiv.org/pdf/2406.19215)
- [Unified Active Retrieval for Retrieval Augmented Generation](https://aclanthology.org/2024.findings-emnlp.999.pdf)

### Tools & Planning

- [Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](https://arxiv.org/pdf/2410.01428)
- [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/pdf/2504.03160)
- [Enhancing Retrieval-Augmented Generation via Evidence Tree Search](https://arxiv.org/pdf/2503.20757)
- [Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs](https://aclanthology.org/2024.findings-emnlp.459.pdf)
- [PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers](https://arxiv.org/pdf/2406.12430)
- [RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval Augmented Generation](https://aclanthology.org/2025.acl-long.179.pdf)
- [RAPID: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery](https://arxiv.org/pdf/2503.00751)

## Training & Trustworthiness

### Preference Alignment

- [PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization](https://arxiv.org/pdf/2412.14510)
- [Towards General Instruction-Following Alignment for Retrieval-Augmented Generation](https://arxiv.org/pdf/2410.09584)

### Knowledge Conflicts & Factuality

- [Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild](https://arxiv.org/pdf/2504.12982)
- [All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment](https://aclanthology.org/2025.findings-acl.588.pdf)
- [Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](https://arxiv.org/pdf/2410.07176v1)
- [Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts When Knowledge Conflicts?](https://arxiv.org/pdf/2401.11911)
- [ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLMs](https://arxiv.org/pdf/2408.12076)
- [FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](https://arxiv.org/pdf/2506.08938)
- [ParamMute: Suppressing Knowledge-Critical FFNs for Faithful Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.15543) · [Code](https://github.com/OpenBMB/ParamMute) · [Project](https://passionate11.github.io/ParamMute-project-page/)
- [Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments](https://arxiv.org/pdf/2406.09815)
- [SPARK: Adaptive Parametric Knowledge Participation for Faithful Retrieval-Augmented Generation](https://arxiv.org/pdf/2606.27786) · [Code](https://github.com/OpenBMB/SPARK)
- [Tug-of-War Between Knowledge: Exploring and Resolving Knowledge Conflicts in Retrieval-Augmented Language Models](https://arxiv.org/pdf/2402.14409)
- [WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia](https://arxiv.org/pdf/2406.13805)

### Security & Privacy

- ["Glue pizza and eat rocks" - Exploiting Vulnerabilities in Retrieval-Augmented Generative Models](https://aclanthology.org/2024.emnlp-main.96.pdf)
- [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)
- [Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection](https://aclanthology.org/2025.findings-acl.1263.pdf)
- [SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](https://arxiv.org/pdf/2501.18636)
- [The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)](https://aclanthology.org/2024.findings-acl.267.pdf)
- The Silent Saboteur: Imperceptible Adversarial Attacks against Black-Box Retrieval-Augmented Generation Systems **[Paper link needed]**
- [TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models](https://arxiv.org/pdf/2405.13401)
- [Typos that Broke the RAG’s Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations](https://aclanthology.org/2024.findings-emnlp.161.pdf)

## Applications & Evaluation

### Specialized Domains & Modalities

- [Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.08826)
- [BSharedRAG: Backbone Shared Retrieval-Augmented Generation for the E-commerce Domain](https://aclanthology.org/2024.findings-emnlp.62.pdf)
- [CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG](https://arxiv.org/pdf/2506.02544)
- [Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion](https://arxiv.org/pdf/2405.19782)
- [EC-RAFT: Automated Generation of Clinical Trial Eligibility Criteria through Retrieval-Augmented Fine-Tuning](https://aclanthology.org/2025.findings-acl.491.pdf)
- [Exploring RAG-based Vulnerability Augmentation with LLMs](https://arxiv.org/pdf/2408.04125)
- [HyKGE: A Hypothesis Knowledge Graph Enhanced RAG Framework for Accurate and Reliable Medical LLMs Responses](https://arxiv.org/pdf/2312.15883)
- [InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales](https://arxiv.org/pdf/2406.13629)
- [Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation](https://arxiv.org/pdf/2501.02226)
- [LTRAG: Enhancing autoformalization and self-refinement for logical reasoning with Thought-Guided RAG](https://openreview.net/pdf?id=6WQZCc9qQ1)
- [Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.04187)
- [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/pdf/2410.13085v1)
- [MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning](https://aclanthology.org/2024.findings-acl.69.pdf)
- [NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](https://arxiv.org/abs/2505.19754)
- [RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records](https://arxiv.org/pdf/2403.00815)
- [Retrieval-Augmented Code Generation for Situated Action Generation: A Case Study on Minecraft](https://aclanthology.org/2024.findings-emnlp.652.pdf)
- [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://aclanthology.org/2024.emnlp-main.62.pdf)
- [TC–RAG: Turing–Complete RAG’s Case study on Medical LLM Systems](https://arxiv.org/pdf/2408.09199)
- [The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit](https://arxiv.org/pdf/2501.02173)
- [Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](https://arxiv.org/pdf/2501.02460)
- [Understanding Retrieval Robustness for Retrieval-augmented Image Captioning](https://arxiv.org/pdf/2406.02265)
- [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://arxiv.org/pdf/2412.14457)

### Evaluation & Analysis

- [Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/pdf/2405.07437)
- MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation **[Paper link needed]**
- [Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.347.pdf)
- [RAG-QA Arena: Evaluating Domain Robustness for Long-form Retrieval Augmented Question Answering](https://arxiv.org/pdf/2407.13998)
- [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.08067)
- [RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](https://arxiv.org/pdf/2408.01262)
- [Searching for Best Practices in Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.981.pdf)
- [Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems](https://aclanthology.org/2024.emnlp-main.552.pdf)
- [Unanswerability Evaluation for Retrieval Augmented Generation](https://arxiv.org/pdf/2412.12300)
- [When Do LLMs Need Retrieval Augmentation? Mitigating LLMs’ Overconfidence Helps Retrieval Augmentation](https://aclanthology.org/2024.findings-acl.675.pdf)

---

## Maintenance notes

- The 15 original categories are retained beneath five navigation groups.
- Each unique paper appears once, based on the inherited primary category. Papers can carry secondary categories in `data/papers.json`.
- Unresolved paper links are shown explicitly as **[Paper link needed]** rather than omitted.
- Code links already present in the source are retained; they have **not** been independently verified in this regeneration.
- Before replacing the live README, synchronize against the latest `main` and review [`data/alignment-report.json`](data/alignment-report.json).
