# RAG: Latest Trends

### TODO
- [x] major categories
- [x] SOTA
- [x] paper links 
- [ ] source codes
This repo aims to understand the latest trends in RAG and LLM.  Contributions welcome!

### Retrieval-Augmented Generation (RAG)

#### Overview

Retrieval-Augmented Generation (RAG) is a powerful paradigm for enhancing large language models (LLMs) by grounding them in external knowledge. This repository provides a categorized collection of recent RAG research, showcasing the evolution from foundational implementations to cutting-edge techniques addressing complex challenges. We cover core concepts, optimization strategies, domain-specific applications, and emerging areas like evaluation, security, and alignment.

## Contents

1.  [**Surveys & Overview Papers**](#1-surveys--overview-papers)
2.  [**Foundational / Classic RAG Models**](#2-foundational--classic-rag-models)
3.  [**Memory-Augmented Models**](#3-memory-augmented-models)
4.  [**Query & Retrieval Optimization**](#4-query--retrieval-optimization)
5.  [**Hybrid & Advanced RAG Architectures**](#5-hybrid--advanced-rag-architectures)
6.  [**Long-Context & Efficiency**](#6-long-context--efficiency)
7.  [**Graph-Based RAG (GraphRAG)**](#7-graph-based-rag-graphrag)
8.  [**Frameworks & Modular Systems**](#8-frameworks--modular-systems)
9.  [**Application-Oriented RAG**](#9-application-oriented-rag)
10. [**Newest Directions (2024–2025)**](#10-newest-directions-20242025)
11. [**Comprehensive RAG research**](#comprehensive-rag-research)

---

## 1. **Surveys & Overview Papers**

*Provides a high-level understanding of the RAG landscape.*

*   Key Takeaways: Big picture & meta-research.
*   [Retrieval-Augmented Generation for Large Language Models: A Survey (2024)](link)
*   [Retrieval-Augmented Generation for Natural Language Processing: A Survey (2024)](link)
*   [Retrieval-Augmented Generation for AI-Generated Content: A Survey (2024)](link)
*   [A Survey on Retrieval-Augmented Text Generation (2022)](link)
*   [Benchmarking LLMs in RAG (2023)](link)
*   [Graph Retrieval-Augmented Generation: A Survey (2024)](link)
*   [Trustworthiness in RAG Systems: A Survey (2024)](link)
*   [Large Language Models and the Future of Information Retrieval (2024)](link)
*   [LLMs for Information Retrieval: A Survey (2023)](link)
*   [The Rise and Potential of LLM-based Agents: A Survey (2023)](link)
*   [Trends in Integration of Knowledge and LLMs: A Survey and Taxonomy (2023)](link)
*   [Hallucination Mitigation in LLMs: A Survey (2024)](link)

## 2. **Foundational / Classic RAG Models**

*Establishes the core RAG paradigm.*

*   Key Takeaways: Early REALM, DPR, RAG architectures.
*   [REALM: Retrieval-Augmented Language Model Pre-Training (2020)](link)
*   [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020)](link)
*   [Dense Passage Retrieval (DPR) for Open-Domain QA (2020)](link)
*   [Entities as Experts: Sparse Memory Access with Entity Supervision (2020)](link)
*   [Generalization through Memorization: kNN-LM (2020)](link)

## 3. **Memory-Augmented Models**

*Focuses on how to improve memory within the LLMs.*

*   Key Takeaways: kNN, memorization techniques.
*   [Memorizing Transformers (2022)](link)
*   [Mention Memory: Incorporating textual knowledge via entity attention (2021)](link)
*   [Training Language Models with Memory Augmentation (2022)](link)
*   [Recitation-Augmented Language Models (2022)](link)
*   [kNN-Prompt: Nearest Neighbor Zero-Shot Inference (2022)](link)
*   [Efficient Nearest Neighbor LMs (2021)](link)

## 4. **Query & Retrieval Optimization**

*Improving retrieval accuracy.*

*   Key Takeaways: Improve the quality of the query.
*   [Learning to Retrieve In-Context Examples for LLMs (2024)](link)
*   [Query Rewriting for RAG (2023)](link)
*   [Retrieve Anything To Augment LLMs (2023)](link)
*   [Reimagining RAG for Answering Queries (2023)](link)
*   [Unsupervised Dense Information Retrieval with Contrastive Learning (2022)](link)
*   [You Can’t Pick Your Neighbors: When and how to rely on retrieval in kNN-LM (2022)](link)

## 5. **Hybrid & Advanced RAG Architectures**

*Novel and creative method to improve RAG performance*

*   Key Takeaways: improve adaptable, efficient and factual generation.
*   [Atlas: Few-shot Learning with RAG (2022)](link)
*   [Active Retrieval-Augmented Generation (2023)](link)
*   [REPLUG: RAG for Black-Box LMs (2023)](link)
*   [RAVEN: In-Context RAG for Encoder-Decoder Models (2023)](link)
*   [In-Context RAG (2023)](link)
*   [InstructRetro: Instruction tuning + RAG pretraining (2023)](link)
*   [Reliable, Adaptable, and Attributable LMs with Retrieval (2024)](link)
*   [Universal Information Extraction with Meta-Pretrained Self-Retrieval (2023)](link)
*   [Pre-computed Memory vs On-the-Fly Encoding: Hybrid RAG (2023)](link)
*   [Neuro-Symbolic Language Modeling with Automaton-Augmented Retrieval (2022)](link)

## 6. **Long-Context & Efficiency**

*Improve RAG performance by scaling for larger content and efficiency*

*   Key Takeaways: Long-range dependencies, long-tail.
*   [Unlimiformer: Long-Range Transformers with Unlimited Input (2023)](link)
*   [Nonparametric Masked Language Modeling (2023)](link)
*   [Improving LMs by Retrieving from Trillions of Tokens (2022)](link)

## 7. **Graph-Based RAG (GraphRAG)**

*Using graph database or graph structure to help the retrieval and reasoning*

*   Key Takeaways: structured knowledge integration
*   [Graph Retrieval-Augmented Generation: A Survey (2024)](link)
*   [Graph-based Summarization: Local to Global RAG (2024)](link)
*   [Graph of Records (GoR): Graphs for Long-Context Summarization (2024)](link)
*   [RAKG: Retrieval-Augmented Knowledge Graph Construction (2025)](link)
*   [DEEP-PolyU GraphRAG curated resources](link)

## 8. **Frameworks & Modular Systems**

*Focus on tooling and make the engineering of RAG simpler and easier*

*   Key Takeaways: practical toolkits, light weight framework, modularity
*   [LightRAG: Simple & Fast RAG (2024)](link)
*   [StructRAG: Hybrid Information Structurization (2024)](link)
*   [RAGLAB: Modular RAG Framework (2024)](link)
*   [FlashRAG: Modular Toolkit (2024)](link)
*   [Speculative RAG: Drafting-based improvements (2024)](link)
*   [InstructRAG: Self-Synthesized Rationales (2024)](link)

## 9. **Application-Oriented RAG**

*Targeting RAG in specific vertical or application*

*   Key Takeaways: target application or domain, better result
*   [PDFTriage: QA over Structured Documents (2023)](link)
*   [WikiChat: Few-Shot Grounding on Wikipedia (2023)](link)
*   [FreshLLMs: Search Engine Augmented (2023)](link)
*   [LeanDojo: Theorem Proving with RAG (2023)](link)
*   [WebGLM: Web-Enhanced QA (2023)](link)
*   [WebCPM: Interactive Web QA (2023)](link)
*   [WebGPT: Browser-Assisted QA (2022)](link)
*   [Teaching LMs to Support Answers with Verified Quotes (2022)](link)

## 10. **Newest Directions (2024–2025)**

*Where is RAG heading*

*   Key Takeaways: The next wave of RAG research.
*   [CReSt: Benchmark for Complex Reasoning with Structured Docs (2025)](link)
*   [RAKG: Retrieval-Augmented Knowledge Graph Construction (2025)](link)
*   [ECoRAG: Evidentiality-guided Compression for Long-Context RAG (2025)](link)
*   [Graph of Records (GoR): Graph-based Long-Context Summarization (2024)](link)

---
## 11. **Comprehensive RAG research**
*   [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)
*   [UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems](https://arxiv.org/pdf/2401.13256)
*   [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/pdf/2401.00396)
*   [M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions](https://arxiv.org/pdf/2405.16420)
*   [Benchmarking Retrieval-Augmented Generation for Medicine](https://aclanthology.org/2024.findings-acl.372.pdf)
*   [To Generate or to Retrieve? On the Effectiveness of Artificial Contexts for Medical Open-Domain Question Answering](https://arxiv.org/pdf/2403.01924)
*    [Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts](https://aclanthology.org/2024.findings-acl.458.pdf)
*   [Synergistic Interplay between Search and Large Language Models for Information Retrieval](https://arxiv.org/pdf/2305.07402)
*   [Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation](https://arxiv.org/pdf/2406.00456)
*   [Chunk-Distilled Language Modeling](https://arxiv.org/pdf/2501.00343)
*   [Landmark Embedding: A Chunking-Free Embedding Method For Retrieval Augmented Long-Context Large Language Models](https://arxiv.org/pdf/2402.11573)
*   [Grounding Language Model with Chunking-Free In-Context Retrieval](https://arxiv.org/pdf/2402.09760)
        *    [Bottleneck-Minimal Indexing for Generative Document Retrieval](https://arxiv.org/pdf/2405.10974)
        *    [Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](https://arxiv.org/pdf/2410.11001)
* [MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System](https://arxiv.org/pdf/2503.09600)
*   [MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2408.17072v1)
*   [Optimizing Query Generation for Enhanced Document Retrieval in RAG](https://arxiv.org/pdf/2407.12325)
*   [BlendFilter: Advancing Retrieval-Augmented Large Language Models via Query Generation Blending and Knowledge Filtering](https://aclanthology.org/2024.emnlp-main.58.pdf)
*   [ARL2: Aligning Retrievers with Black-box Large Language Models via Self-guided Adaptive Relevance Labeling](https://arxiv.org/pdf/2402.13542)
*   [RaFe: Ranking Feedback Improves Query Rewriting for RAG](https://aclanthology.org/2024.findings-emnlp.49.pdf)
* [Diversify-verify-adapt: Efficient and Robust Retrieval-Augmented Ambiguous Question Answering](https://arxiv.org/pdf/2409.02361)
* [UniRAG: Unified Query Understanding Method for Retrieval Augmented Generation](https://openreview.net/attachment?id=h68SaHDtal&name=pdf)
*   [EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.12559)
*   [xRAG: Extreme Context Compression for Retrieval-augmented Generation with One Token](https://arxiv.org/pdf/2405.13792)
*   [FIT-RAG: Black-Box RAG with Factual Information and Token Reduction](https://arxiv.org/pdf/2403.14374)
*   [Making Retrieval-Augmented Language Models Robust to Irrelevant Context](https://arxiv.org/pdf/2310.01558)
*   [An Information Bottleneck Perspective for Effective Noise Filtering on Retrieval-Augmented Generation](https://arxiv.org/pdf/2406.01549)
*   [Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Handling Noisy Contexts](https://aclanthology.org/2024.findings-emnlp.136.pdf)
*   [Small Models, Big Insights: Leveraging Slim Proxy Models To Decide When and What to Retrieve for LLMs](https://arxiv.org/pdf/2402.12052)
*   [Provence: efficient and robust context pruning for retrieval-augmented generation](https://arxiv.org/pdf/2501.16214)
        *    [Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection](https://arxiv.org/pdf/2405.16178)
        *    [PISCO: Pretty Simple Compression for Retrieval-Augmented Generation](https://arxiv.org/pdf/2501.16075)
        *    [ECoRAG: Evidentiality-guided Compression for Long Context RAG](https://arxiv.org/pdf/2506.05167)
*   [Hierarchical Retrieval-Augmented Generation Model with Rethink for Multi-hop Question Answering](https://arxiv.org/abs/2408.11875)
*   [EfficientRAG: Efficient Retriever for Multi-Hop Question Answering](https://arxiv.org/pdf/2408.04259)
*   [Retrieve, Summarize, Plan: Advancing Multi-hop Question Answering with an Iterative Approach](https://arxiv.org/pdf/2407.13101)
*   [Generate-then-Ground in Retrieval-Augmented Generation for Multi-hop Question Answering](https://arxiv.org/pdf/2406.14891)
*   [REANO: Optimising Retrieval-Augmented Reader Models through Knowledge Graph Generation](https://aclanthology.org/2024.acl-long.115.pdf)
* [SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.15272)
* [HopRAG: Multi-Hop Reasoning for Logic-Aware Retrieval Augmented Generation](https://arxiv.org/pdf/2502.12442)
* [GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model Reasoning on Knowledge Graphs](https://arxiv.org/pdf/2405.20139)
* [Retrieve, Summarize, Plan: Advancing Multi-hop Question Answering with an Iterative Approach](https://arxiv.org/pdf/2407.13101)
* [HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/pdf/2503.10150)
* [RASPberry: Retrieval-Augmented Monte Carlo Tree Self-Play with Reasoning Consistency for Multi-Hop Question Answering](https://aclanthology.org/2025.findings-acl.587.pdf)
* [FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs](https://arxiv.org/pdf/2501.09957)
*  [Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG and Symbolic Verification](https://arxiv.org/pdf/2504.04578)
*   [Tug-of-War Between Knowledge: Exploring and Resolving Knowledge Conflicts in Retrieval-Augmented Language Models](https://arxiv.org/pdf/2402.14409)
*   [Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts When Knowledge Conflicts?](https://arxiv.org/pdf/2401.11911)
*   [Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](https://arxiv.org/pdf/2410.07176v1)
*   [Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments](https://arxiv.org/pdf/2406.09815)
* [FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](https://arxiv.org/pdf/2506.08938)
* [Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild](https://arxiv.org/pdf/2504.12982)
*  [WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia](https://arxiv.org/pdf/2406.13805)
        *   [ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLMs](https://arxiv.org/pdf/2408.12076)
*  [All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment](https://aclanthology.org/2025.findings-acl.588.pdf)
*   [Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.04187)
*   [RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models](https://aclanthology.org/2024.emnlp-main.62.pdf)
*   [Exploring RAG-based Vulnerability Augmentation with LLMs](https://arxiv.org/pdf/2408.04125)
*   [RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records](https://arxiv.org/pdf/2403.00815)
*   [Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion](https://arxiv.org/pdf/2405.19782)
*   [Understanding Retrieval Robustness for Retrieval-augmented Image Captioning](https://arxiv.org/pdf/2406.02265)
*   [MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models](https://arxiv.org/pdf/2410.13085v1)
*   [MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning](https://aclanthology.org/2024.findings-acl.69.pdf)
*   [Ask in Any Modality: A Comprehensive Survey on Multimodal Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.08826)
*   [BSharedRAG: Backbone Shared Retrieval-Augmented Generation for the E-commerce Domain](https://aclanthology.org/2024.findings-emnlp.62.pdf)
        *   [TC–RAG: Turing–Complete RAG’s Case study on Medical LLM Systems](https://arxiv.org/pdf/2408.09199)
        *   [Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation](https://arxiv.org/pdf/2501.02226)
        *   [VISA: Retrieval Augmented Generation with Visual Source Attribution](https://arxiv.org/pdf/2412.14457)
        *   [The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit](https://arxiv.org/pdf/2501.02173)
        *   [HyKGE: A Hypothesis Knowledge Graph Enhanced RAG Framework for Accurate and Reliable Medical LLMs Responses](https://arxiv.org/pdf/2312.15883)
        *   [NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](https://arxiv.org/abs/2505.19754)
        *   [CoRe-MMRAG: Cross-Source Knowledge Reconciliation for Multimodal RAG](https://arxiv.org/pdf/2506.02544)
        *   [Retrieval-Augmented Code Generation for Situated Action Generation: A Case Study on Minecraft](https://aclanthology.org/2024.findings-emnlp.652.pdf)
        *   [InstructRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales](https://arxiv.org/pdf/2406.13629)
        *   [Towards Omni-RAG: Comprehensive Retrieval-Augmented Generation for Large Language Models in Medical Applications](https://arxiv.org/pdf/2501.02460)
        *   [EC-RAFT: Automated Generation of Clinical Trial Eligibility Criteria through Retrieval-Augmented Fine-Tuning](https://aclanthology.org/2025.findings-acl.491.pdf)
        *   [LTRAG: Enhancing autoformalization and self-refinement for logical reasoning with Thought-Guided RAG](https://openreview.net/pdf?id=6WQZCc9qQ1)
*   [DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models](https://arxiv.org/pdf/2403.10081)
*   [RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback](https://aclanthology.org/2024.findings-acl.281.pdf)
       *SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](https://arxiv.org/pdf/2410.13553v2)
*   [Retrieving, Rethinking and Revising: The Chain-of-Verification Can Improve Retrieval Augmented Generation](https://aclanthology.org/2024.findings-emnlp.607.pdf)
*   [KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.18397)
*   [Retrieval-Augmented Generation by Evidence Retroactivity in LLMs](https://arxiv.org/pdf/2501.05475)
        *   [Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience](https://arxiv.org/pdf/2506.00842)
  * [ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation](https://arxiv.org/pdf/2503.21729v1)
        *   [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.05312)
*   [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/pdf/2406.15319)
*   [LongRAG: A Dual-perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering](https://aclanthology.org/2024.emnlp-main.1259.pdf)
*   [On the Role of Long-tail Knowledge in Retrieval Augmented Large Language Models](https://arxiv.org/pdf/2406.16367)
*   [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.17031)
*   [Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG](https://arxiv.org/pdf/2410.05983)
        *    [Inference Scaling for Long-Context Retrieval Augmented Generation](https://arxiv.org/pdf/2410.04343)
         * [Long-Context Inference with Retrieval-Augmented Speculative Decoding](https://arxiv.org/pdf/2502.20330)
*   [BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack](https://arxiv.org/pdf/2406.10149)
* [Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](https://arxiv.org/pdf/2505.10413)
*   [ACTIVERAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/pdf/2402.13547)
        *  [SEAKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](https://arxiv.org/pdf/2406.19215)
        *    [Unified Active Retrieval for Retrieval Augmented Generation](https://aclanthology.org/2024.findings-emnlp.999.pdf)
*     [Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](https://arxiv.org/pdf/2505.12731)
* [DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.10198)
*   [PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers](https://arxiv.org/pdf/2406.12430)
   *Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs(https://aclanthology.org/2024.findings-emnlp.459.pdf)
   * [RAPID: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery](https://arxiv.org/pdf/2503.00751)
*  [RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval Augmented Generation](https://aclanthology.org/2025.acl-long.179.pdf)
   * [Enhancing Retrieval-Augmented Generation via Evidence Tree Search](https://arxiv.org/pdf/2503.20757)
* [Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](https://arxiv.org/pdf/2410.01428)
*  [Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG and Symbolic Verification](https://arxiv.org/pdf/2504.04578)
* [DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments](https://arxiv.org/pdf/2504.03160)
*   [OpenRAG: Enhanced Retrieval Augmented Reasoning with Open-Source Large Language Models](https://arxiv.org/pdf/2410.01782)
*   [RAGLAB: A Modular and Research-Oriented Unified Framework for Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.11381)
*   [From RAG to RICHES: Retrieval Interlaced with Sequence Generation](https://arxiv.org/pdf/2407.00361)
*   [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/pdf/2310.11511)
*   [UltraRAG: A Modular and Automated Toolkit for Adaptive Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.08761)
*   [GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](https://arxiv.org/pdf/2505.18710)
        *   [Parametric Retrieval Augmented Generation](https://arxiv.org/pdf/2501.15915)
        *    [RAGraph: A General Retrieval-Augmented Graph Learning Framework](https://arxiv.org/pdf/2410.23855)
        *    [FRAG: A Flexible Modular Framework for Retrieval-Augmented Generation based on Knowledge Graphs](https://arxiv.org/pdf/2501.09957)
         * [ClueAnchor: Clue-Anchored Knowledge Reasoning Exploration and Optimization for Retrieval-Augmented Generation](https://arxiv.org/pdf/2505.24388)
*  [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.05312)
*   [ComposeRAG: A Modular and Composable RAG for Corpus-Grounded Multi-Hop Question Answering](https://arxiv.org/pdf/2506.00232)
*   [MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](https://arxiv.org/pdf/2501.00332)
*   [Towards General Instruction-Following Alignment for Retrieval-Augmented Generation](https://arxiv.org/pdf/2410.09584)
*   [PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization](https://arxiv.org/pdf/2412.14510)
*   [All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment](https://aclanthology.org/2025.findings-acl.588.pdf)
*  [GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](https://arxiv.org/pdf/2505.18710)
*   [The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)](https://aclanthology.org/2024.findings-acl.267.pdf)
*   ["Glue pizza and eat rocks" - Exploiting Vulnerabilities in Retrieval-Augmented Generative Models](https://aclanthology.org/2024.emnlp-main.96.pdf)
*   [Typos that Broke the RAG’s Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations](https://aclanthology.org/2024.findings-emnlp.161.pdf)
*   [SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](https://arxiv.org/pdf/2501.18636)
   * [TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models](https://arxiv.org/pdf/2405.13401)
* Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection ([https://aclanthology.org/2025.findings-acl.1263.pdf](https://aclanthology.org/2025.findings-acl.1263.pdf))
*  [The Silent Saboteur: Imperceptible Adversarial Attacks against Black-Box Retrieval-Augmented Generation Systems]()
*  [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)
*   [Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/pdf/2405.07437)
*   [RAG-QA Arena: Evaluating Domain Robustness for Long-form Retrieval Augmented Question Answering](https://arxiv.org/pdf/2407.13998)
*   [Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](https://arxiv.org/pdf/2410.07176v1)
*   [When Do LLMs Need Retrieval Augmentation? Mitigating LLMs’ Overconfidence Helps Retrieval Augmentation](https://aclanthology.org/2024.findings-acl.675.pdf)
*   [Searching for Best Practices in Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.981.pdf)
*   [Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.347.pdf)
    *   [Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems](https://aclanthology.org/2024.emnlp-main.552.pdf)
    *   [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](https://arxiv.org/pdf/241

## 11. **Comprehensive RAG Research (Chronological Order)**

This section presents a comprehensive list of RAG research papers, organized chronologically to illustrate the evolution of the field. Key conferences like ACL and EMNLP are noted.

### 2023

*   [Synergistic Interplay between Search and Large Language Models for Information Retrieval](https://arxiv.org/pdf/2305.07402)
*    [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/pdf/2403.10131?trk=public_post_comment-text) (potentially 2024 even if submitted in 2023, recheck)
### 2024

*   [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)
*   [UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems](https://arxiv.org/pdf/2401.13256)
*   [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/pdf/2401.00396)
*   [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.17031)
*   [Landmark Embedding: A Chunking-Free Embedding Method For Retrieval Augmented Long-Context Large Language Models](https://arxiv.org/pdf/2402.11573)
*    [G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering](https://arxiv.org/pdf/2402.07630)
*   [Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts When Knowledge Conflicts?](https://arxiv.org/pdf/2401.11911)
*    [Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts](https://aclanthology.org/2024.findings-acl.458.pdf) (ACL Findings)
*    [BSharedRAG: Backbone Shared Retrieval-Augmented Generation for the E-commerce Domain](https://aclanthology.org/2024.findings-emnlp.62.pdf) (EMNLP Findings)
* [Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Handling Noisy Contexts](https://arxiv.org/pdf/2408.01084)
*    [An Information Bottleneck Perspective for Effective Noise Filtering on Retrieval-Augmented Generation](https://arxiv.org/pdf/2406.01549)
*     [An Information Bottleneck Perspective for Effective Noise Filtering on Retrieval-Augmented Generation](https://arxiv.org/pdf/2406.01549)
*   [Multi-Head RAG: Solving Multi-Aspect Problems with LLMs](https://arxiv.org/pdf/2406.05085?trk=public_post_comment-text)
*    [Multi-Head RAG: Solving Multi-Aspect Problems with LLMs](https://arxiv.org/pdf/2406.05085?trk=public_post_comment-text)
*   [When Do LLMs Need Retrieval Augmentation? Mitigating LLMs’ Overconfidence Helps Retrieval Augmentation](https://aclanthology.org/2024.findings-acl.675.pdf) (ACL Findings)
*   [Exploring RAG-based Vulnerability Augmentation with LLMs](https://arxiv.org/pdf/2408.04125)
*   [Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.04187)
*   [Landmark Embedding: A Chunking-Free Embedding Method For Retrieval Augmented Long-Context Large Language Models](https://arxiv.org/pdf/2402.11573)
*   [BlendFilter: Advancing Retrieval-Augmented Large Language Models via Query Generation Blending and Knowledge Filtering](https://aclanthology.org/2024.emnlp-main.58.pdf) (EMNLP Main)
*   [MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2408.17072v1)
* [RAG-QA Arena: Evaluating Domain Robustness for Long-form Retrieval Augmented Question Answering](https://arxiv.org/pdf/2407.13998)
*  [Retrieve, Summarize, Plan: Advancing Multi-hop Question Answering with an Iterative Approach](https://arxiv.org/pdf/2407.13101)
* [EfficientRAG: Efficient Retriever for Multi-Hop Question Answering](https://arxiv.org/pdf/2408.04259)
*   [Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Handling Noisy Contexts](https://arxiv.org/pdf/2408.01084)
*   [REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering](https://aclanthology.org/2024.emnlp-main.321.pdf)
*   [EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.12559)
*   [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/pdf/2406.15319)
*    [Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](https://arxiv.org/pdf/2410.07176v1)
*    [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.08067)
*   [StructuredRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-Time Hybrid Information Structurization](https://arxiv.org/pdf/2410.08815)
*   [PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization](https://arxiv.org/pdf/2412.14510)
*   [TC–RAG: Turing–Complete RAG’s Case study on Medical LLM Systems](https://arxiv.org/pdf/2408.09199)
### 2025
*   [A Multi-Task Embedder For Retrieval Augmented LLM](https://aclanthology.org/2024.acl-long.194.pdf)
*   [Chunk-Distilled Language Modeling](https://arxiv.org/pdf/2501.00343)
*   [All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment](https://aclanthology.org/2025.findings-acl.588.pdf) (ACL Findings)
*   [Enhancing Retrieval-Augmented Generation via Evidence Tree Search](https://arxiv.org/pdf/2503.20757)
*  [Long-Context Inference with Retrieval-Augmented Speculative Decoding](https://arxiv.org/pdf/2502.20330)
*  [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.05312)
*   [Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering](https://arxiv.org/pdf/2502.14245)
*   [HiRAG: Retrieval-Augmented Generation with Hierarchical Knowledge](https://arxiv.org/pdf/2503.10150)
*   [Medical Graph RAG: Evidence-based Medical Large Language Model via Graph Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.04187)
*   [NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](https://arxiv.org/abs/2505.19754)
*    [On the Robustness of RAG Systems in Educational Question Answering under Knowledge Discrepancies](https://aclanthology.org/2025.acl-short.16.pdf)
*  [Parametric Retrieval Augmented Generation](https://arxiv.org/pdf/2501.15915)
*  [ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation](https://arxiv.org/pdf/2503.21729v1)
*  [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.05312)
*   [UltraRAG: A Modular and Automated Toolkit for Adaptive Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.08761)
*  [GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](https://arxiv.org/pdf/2505.18710)
*   [StructuredRAG: Boosting Knowledge Intensive Reasoning of LLMs VIA INFERENCE-TIME HYBRID INFORMATION STRUCTURIZATION](https://arxiv.org/pdf/2410.08815)/