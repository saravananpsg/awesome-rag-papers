# RAG: Latest trends
### TODO
- [x] major categories
- [ ] SOTA
- [ ] paper links 
- [ ] source codes
This repo is understand the latest trends of RAG and LLM

### Retrieval-Augmented Generation (RAG)
I am very sorry, this is the full list based on the list you provided, please note some could be based on my understanding, please re-check to make sure that it match your expectation and your domain knowledge.

### Chapter X: A Chronological Exploration of Retrieval-Augmented Generation Research

#### Overview

This chapter provides a structured journey through the evolution of Retrieval-Augmented Generation (RAG) research. By categorizing and presenting key papers chronologically, we aim to illustrate the progression of ideas, from foundational RAG implementations to cutting-edge techniques addressing complex challenges. This chapter will help you understand not only *what* the latest RAG methods are but also *why* they emerged and how they build upon previous work. We will cover the core concepts, optimization strategies, domain-specific applications, and the critical emerging areas of evaluation, security, and alignment. This will equip you with a well-rounded understanding of the RAG landscape.

#### I. Naive RAG: The Fundamentals of Retrieval and Generation

*   **Description:** This section covers the essential building blocks of RAG: retrieving relevant information and incorporating it into the generation process. These papers demonstrate the initial attempts to combine external knowledge with language models, addressing the core problem of knowledge augmentation. While these methods might seem simple compared to modern techniques, they establish the foundation for all subsequent RAG research.
*   **Key Takeaways:**
    *   RAG fundamentally combines retrieval and generation for enhanced language model performance.
    *   Early research focused on proving the basic concept and identifying its potential.
    *   Hallucination and limited knowledge were early challenges that motivated further innovation.
*   **Papers:**
    *   [Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)
    *   [UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems](https://arxiv.org/pdf/2401.13256)
    *   [RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models](https://arxiv.org/pdf/2401.00396)
    *   [M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions](https://arxiv.org/pdf/2405.16420)
    *   [Benchmarking Retrieval-Augmented Generation for Medicine](https://aclanthology.org/2024.findings-acl.372.pdf)
    *   [To Generate or to Retrieve? On the Effectiveness of Artificial Contexts for Medical Open-Domain Question Answering](https://arxiv.org/pdf/2403.01924)
    *    [Improving Retrieval Augmented Open-Domain Question-Answering with Vectorized Contexts](https://aclanthology.org/2024.findings-acl.458.pdf)
    *   [Synergistic Interplay between Search and Large Language Models for Information Retrieval](https://arxiv.org/pdf/2305.07402)

#### II. Chunking/Indexing Strategies: Optimizing Information Access

*   **Description:** This section explores the crucial role of document chunking and indexing in RAG systems. These papers investigate different methods for dividing documents into manageable segments, creating efficient indexes for retrieval, and ensuring that relevant information can be accessed quickly and accurately. The effectiveness of RAG heavily relies on these preprocessing steps.
*   **Key Takeaways:**
    *   Effective chunking is essential for capturing relevant context while minimizing noise.
    *   Indexing methods directly impact retrieval speed and accuracy.
    *   Different document types and tasks might require tailored chunking and indexing strategies.
*   **Papers:**
    *   [Mix-of-Granularity: Optimize the Chunking Granularity for Retrieval-Augmented Generation](https://arxiv.org/pdf/2406.00456)
    *   [Chunk-Distilled Language Modeling](https://arxiv.org/pdf/2501.00343)
    *   [Landmark Embedding: A Chunking-Free Embedding Method For Retrieval Augmented Long-Context Large Language Models](https://arxiv.org/pdf/2402.11573)
    *   [Grounding Language Model with Chunking-Free In-Context Retrieval](https://arxiv.org/pdf/2402.09760)
        *    [Bottleneck-Minimal Indexing for Generative Document Retrieval](https://arxiv.org/pdf/2405.10974)
        *    [Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](https://arxiv.org/pdf/2410.11001)
* [MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System](https://arxiv.org/pdf/2503.09600)

#### III. Query Enhancement & Relevance Improvement

*   **Description:** This section focuses on techniques for improving the quality of the query used for retrieval. This includes query rewriting, expansion, and methods to better align the query with the user's information need and the language model's processing capabilities. Improving query quality is a critical step towards retrieving more relevant and useful context.
*   **Key Takeaways:**
    *   Simple user queries are often insufficient for effective retrieval.
    *   Query rewriting and expansion can significantly improve retrieval accuracy.
    *   Aligning the query with the LLM's internal representations can further enhance relevance.
*   **Papers:**
    *   [MaFeRw: Query Rewriting with Multi-Aspect Feedbacks for Retrieval-Augmented Large Language Models](https://arxiv.org/pdf/2408.17072v1)
    *   [Optimizing Query Generation for Enhanced Document Retrieval in RAG](https://arxiv.org/pdf/2407.12325)
    *   [BlendFilter: Advancing Retrieval-Augmented Large Language Models via Query Generation Blending and Knowledge Filtering](https://aclanthology.org/2024.emnlp-main.58.pdf)
    *   [ARL2: Aligning Retrievers with Black-box Large Language Models via Self-guided Adaptive Relevance Labeling](https://arxiv.org/pdf/2402.13542)
    *   [RaFe: Ranking Feedback Improves Query Rewriting for RAG](https://aclanthology.org/2024.findings-emnlp.49.pdf)
* [Diversify-verify-adapt: Efficient and Robust Retrieval-Augmented Ambiguous Question Answering](https://arxiv.org/pdf/2409.02361)
* [UniRAG: Unified Query Understanding Method for Retrieval Augmented Generation](https://openreview.net/attachment?id=h68SaHDtal&name=pdf)

#### IV. Context Optimization & Selection

*   **Description:** This section delves into the methods for dealing with the retrieved documents. This includes context compression, filtering irrelevant or noisy information, and strategically selecting the most valuable context to pass to the LLM. The goal is to minimize distractions and focus the language model on the most pertinent information.
*   **Key Takeaways:**
    *   Retrieved documents often contain irrelevant or redundant information.
    *   Context compression techniques can reduce computational cost and improve generation quality.
    *   Selective context inclusion helps the LLM focus on the most important information.
*   **Papers:**
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

#### V. Multi-Hop Reasoning & Knowledge Graphs

*   **Description:** This section explores techniques that enable RAG systems to handle complex queries requiring reasoning across multiple documents or knowledge sources. These papers focus on multi-hop question answering and the integration of knowledge graphs to enhance the reasoning capabilities of RAG systems.
*   **Key Takeaways:**
    *   Complex questions often require synthesizing information from multiple sources.
    *   Knowledge graphs can provide structured knowledge to aid reasoning.
    *   Multi-hop retrieval and reasoning demand more sophisticated RAG architectures.
*   **Papers:**
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

#### VI. Knowledge Conflicts & Factuality

*   **Description:** This section addresses the critical challenge of knowledge conflicts, where retrieved information contradicts existing knowledge or other retrieved sources. The papers in this section explore methods for detecting and resolving these conflicts to ensure the factuality and reliability of the generated outputs.
*   **Key Takeaways:**
    *   RAG systems must be robust to conflicting information from external sources.
    *   Knowledge conflict detection is essential for maintaining factuality.
    *   Resolution strategies may involve ranking sources, fact-checking, or modifying the generation process.
*   **Papers:**
    *   [Tug-of-War Between Knowledge: Exploring and Resolving Knowledge Conflicts in Retrieval-Augmented Language Models](https://arxiv.org/pdf/2402.14409)
    *   [Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts When Knowledge Conflicts?](https://arxiv.org/pdf/2401.11911)
    *   [Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](https://arxiv.org/pdf/2410.07176v1)
    *   [Retrieval Augmented Fact Verification by Synthesizing Contrastive Arguments](https://arxiv.org/pdf/2406.09815)
* [FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation](https://arxiv.org/pdf/2506.08938)
* [Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild](https://arxiv.org/pdf/2504.12982)
*  [WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia](https://arxiv.org/pdf/2406.13805)
        *   [ConflictBank: A Benchmark for Evaluating the Influence of Knowledge Conflicts in LLMs](https://arxiv.org/pdf/2408.12076)
*  [All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment](https://aclanthology.org/2025.findings-acl.588.pdf)

#### VII. RAG for Specialized Domains & Modalities

*   **Description:** This section showcases the application of RAG techniques in specific domains, such as medicine, code generation, and e-commerce, and explores the integration of multiple modalities, including text, images, and audio. These papers demonstrate the versatility of RAG and its potential to enhance language models in diverse contexts.
*   **Key Takeaways:**
    *   RAG can be adapted and optimized for various domains and tasks.
    *   Integrating multiple modalities expands the scope and applicability of RAG.
    *   Domain-specific knowledge and data are crucial for effective RAG implementation.
*   **Papers:**
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

#### VIII. Iterative Retrieval & Reasoning

*   **Description:** This section explores techniques that enable RAG systems to refine their knowledge over time through iterative retrieval and reasoning. By dynamically adjusting retrieval strategies based on evolving information needs and incorporating feedback loops, these methods aim to improve the accuracy and relevance of generated outputs.
*   **Key Takeaways:**
    *   Iterative retrieval allows RAG systems to adapt to complex and evolving information needs.
    *   Feedback mechanisms enable continuous learning and refinement of knowledge.
    *   Dynamic retrieval strategies improve the accuracy and relevance of generated outputs.
*   **Papers:**
    *   [DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models](https://arxiv.org/pdf/2403.10081)
    *   [RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback](https://aclanthology.org/2024.findings-acl.281.pdf)
       *SynapticRAG: Enhancing Temporal Memory Retrieval in Large Language Models through Synaptic Mechanisms](https://arxiv.org/pdf/2410.13553v2)
*   [Retrieving, Rethinking and Revising: The Chain-of-Verification Can Improve Retrieval Augmented Generation](https://aclanthology.org/2024.findings-emnlp.607.pdf)
*   [KiRAG: Knowledge-Driven Iterative Retriever for Enhancing Retrieval-Augmented Generation](https://arxiv.org/pdf/2502.18397)
*   [Retrieval-Augmented Generation by Evidence Retroactivity in LLMs](https://arxiv.org/pdf/2501.05475)
        *   [Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience](https://arxiv.org/pdf/2506.00842)
  * [ReaRAG: Knowledge-guided Reasoning Enhances Factuality of Large Reasoning Models with Iterative Retrieval Augmented Generation](https://arxiv.org/pdf/2503.21729v1)
        *   [Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.05312)

#### IX. Long Context RAG

*   **Description:** This section discusses the challenges and opportunities of scaling RAG systems to handle long contexts, allowing them to process and synthesize information from extensive documents and knowledge sources. These papers explore techniques for managing long-range dependencies and improving the efficiency of retrieval and generation in long-context RAG systems.
*   **Key Takeaways:**
    *   Long contexts enable RAG systems to process and synthesize more extensive information.
    *   Managing long-range dependencies is essential for accurate and coherent generation.
    *   Efficient retrieval and generation techniques are critical for scaling RAG to long contexts.
*   **Papers:**
    *   [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/pdf/2406.15319)
    *   [LongRAG: A Dual-perspective Retrieval-Augmented Generation Paradigm for Long-Context Question Answering](https://aclanthology.org/2024.emnlp-main.1259.pdf)
    *   [On the Role of Long-tail Knowledge in Retrieval Augmented Large Language Models](https://arxiv.org/pdf/2406.16367)
    *   [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.17031)
    *   [Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG](https://arxiv.org/pdf/2410.05983)
        *    [Inference Scaling for Long-Context Retrieval Augmented Generation](https://arxiv.org/pdf/2410.04343)
         * [Long-Context Inference with Retrieval-Augmented Speculative Decoding](https://arxiv.org/pdf/2502.20330)
*   [BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack](https://arxiv.org/pdf/2406.10149)
* [Hierarchical Document Refinement for Long-context Retrieval-augmented Generation](https://arxiv.org/pdf/2505.10413)

#### X. Active Learning and Adaptive Retrieval

*   **Description:** This section explores techniques that enable RAG systems to learn which documents to retrieve based on active learning principles. By actively selecting the most informative documents for retrieval, these methods aim to improve the efficiency and accuracy of RAG systems.
*   **Key Takeaways:**
    *   Active learning enables RAG systems to learn which documents to retrieve.
    *   Adaptive retrieval strategies improve efficiency and accuracy.
*   **Papers:**
    *   [ACTIVERAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/pdf/2402.13547)
        *  [SEAKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](https://arxiv.org/pdf/2406.19215)
        *    [Unified Active Retrieval for Retrieval Augmented Generation](https://aclanthology.org/2024.findings-emnlp.999.pdf)
*     [Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](https://arxiv.org/pdf/2505.12731)
* [DioR: Adaptive Cognitive Detection and Contextual Retrieval Optimization for Dynamic Retrieval-Augmented Generation](https://arxiv.org/pdf/2504.10198)

#### XI. Integration of External Tools & Planning

*   **Description:** This section explores approaches that involve integrating external tools or planning stages into the RAG process. This expands the capabilities of RAG beyond simple retrieval and generation, allowing it to tackle more complex tasks.
*   **Key Takeaways:**
    *   External tools can augment RAG systems with specialized functionalities.
    *   Planning can improve the coherence and effectiveness of RAG for complex tasks.
*   **Papers:**
    *   [PlanRAG: A Plan-then-Retrieval Augmented Generation for Generative Large Language Models as Decision Makers](https://arxiv.org/pdf/2406.12430)
   *Learning to Plan for Retrieval-Augmented Large Language Models from Knowledge Graphs(https://aclanthology.org/2024.findings-emnlp.459.pdf)
   * [RAPID: Efficient Retrieval-Augmented Long Text Generation with Writing Planning and Information Discovery](https://arxiv.org/pdf/2503.00751)
*  [RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval Augmented Generation](https://aclanthology.org/2025.acl-long.179.pdf)
   * [Enhancing Retrieval-Augmented Generation via Evidence Tree Search](https://arxiv.org/pdf/2503.20757)
* [Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](https://arxiv.org/pdf/2410.01428)
*  [Hierarchical Planning for Complex Tasks with Knowledge Graph-RAG and Symbolic Verification](https://arxiv.org/pdf/2504.04578)
* DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments ([https://arxiv.org/pdf/2504.03160](https://arxiv.org/pdf/2504.03160))

#### XII. Advanced RAG Architectures & Frameworks (Trending)

*   **Description:** These papers present more sophisticated RAG pipelines, often involving multiple steps, agents, planning, or learned components. They represent current research directions that move beyond the basic "retrieve and generate" paradigm.
*   **Key Takeaways:**
    *   RAG systems are becoming increasingly complex and modular.
    *   Advanced architectures aim to improve adaptability, efficiency, and factuality.
*   **Papers:**
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

#### XIII. Preference Alignment

*   **Description:** As RAG systems become more sophisticated, aligning their behavior with human preferences is crucial. These papers explore methods for training RAG systems to generate outputs that are not only accurate but also helpful, harmless, and aligned with human values.
*   **Key Takeaways:**
    *   Aligning RAG systems with human preferences is essential for real-world deployment.
    *   Preference alignment techniques can improve the helpfulness and harmlessness of RAG outputs.
*   **Papers:**
    *   [Towards General Instruction-Following Alignment for Retrieval-Augmented Generation](https://arxiv.org/pdf/2410.09584)
    *   [PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization](https://arxiv.org/pdf/2412.14510)
*   [All That Glitters is Not Gold: Improving Robust Retrieval-Augmented Language Models with Fact-Centric Preference Alignment](https://aclanthology.org/2025.findings-acl.588.pdf)
*  [GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](https://arxiv.org/pdf/2505.18710)

#### XIV. Security & Privacy of RAG

*   **Description:** This section highlights the emerging concerns about security vulnerabilities and privacy risks in RAG systems. These papers explore potential attack vectors and propose defense mechanisms to protect RAG systems from malicious actors and ensure user privacy.
*   **Key Takeaways:**
    *   RAG systems can be vulnerable to adversarial attacks and data breaches.
    *   Security and privacy considerations are essential for responsible RAG development.
*   **Papers:**
    *   [The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG)](https://aclanthology.org/2024.findings-acl.267.pdf)
    *   ["Glue pizza and eat rocks" - Exploiting Vulnerabilities in Retrieval-Augmented Generative Models](https://aclanthology.org/2024.emnlp-main.96.pdf)
    *   [Typos that Broke the RAG’s Back: Genetic Attack on RAG Pipeline by Simulating Documents in the Wild via Low-level Perturbations](https://aclanthology.org/2024.findings-emnlp.161.pdf)
*   [SafeRAG: Benchmarking Security in Retrieval-Augmented Generation of Large Language Model](https://arxiv.org/pdf/2501.18636)
   * [TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models](https://arxiv.org/pdf/2405.13401)
* Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection ([https://aclanthology.org/2025.findings-acl.1263.pdf](https://aclanthology.org/2025.findings-acl.1263.pdf))
*  [The Silent Saboteur: Imperceptible Adversarial Attacks against Black-Box Retrieval-Augmented Generation Systems]()
*  [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)

#### XV. Evaluation & Analysis of RAG (Trending)

*   **Description:** This section is devoted to papers that analyze and evaluate the effectiveness of RAG systems, identify their limitations, and develop better evaluation metrics and benchmarks. These efforts are essential for driving progress in the field and ensuring that RAG systems are reliable and trustworthy.
*   **Key Takeaways:**
    *   Rigorous evaluation is critical for understanding the strengths and weaknesses of RAG systems.
    *   Standardized benchmarks and metrics enable fair comparisons between different approaches.
*   **Papers:**
    *   [Evaluation of Retrieval-Augmented Generation: A Survey](https://arxiv.org/pdf/2405.07437)
    *   [RAG-QA Arena: Evaluating Domain Robustness for Long-form Retrieval Augmented Question Answering](https://arxiv.org/pdf/2407.13998)
    *   [Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](https://arxiv.org/pdf/2410.07176v1)
    *   [When Do LLMs Need Retrieval Augmentation? Mitigating LLMs’ Overconfidence Helps Retrieval Augmentation](https://aclanthology.org/2024.findings-acl.675.pdf)
    *   [Searching for Best Practices in Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.981.pdf)
*   [Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation](https://aclanthology.org/2024.emnlp-main.347.pdf)
    *   [Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems](https://aclanthology.org/2024.emnlp-main.552.pdf)
    *   [A Reality Check on Context Utilisation for Retrieval-Augmented Generation](https://arxiv.org/pdf/2412.17031)
   * [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://arxiv.org/pdf/2408.08067)
* [RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework](https://arxiv.org/pdf/2408.01262)
* [Unanswerability Evaluation for Retrieval Augmented Generation](https://arxiv.org/pdf/2412.12300)
* [MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](https://arxiv.org/pdf/250



### Exploring the LLMs with RAGs(In progress)

### Foundational Paradigms: 
The authors categorize RAG methods into four paradigms based on how the retriever augments the generator:
 * **Query-based RAG**: Retrieval results are used to augment the input of the generative model.
 * **Latent Representation-based RAG**: Retrieved information is integrated into the hidden layers of the generator.
 * **Logit-based RAG**: Information retrieval influences the logits (probabilities) during the generative process.
 * **Speculative RAG**: The model uses retrieval to avoid unnecessary generation steps, saving computational resources.

## Hybrid Search Approaches
* Combining dense and sparse retrieval methods for more accurate and diverse results 
* Integration of semantic search with keyword-based approaches

## Multi-Modal RAG
* Extending RAG to handle not just text, but also images, audio, and video
* Improved context understanding through multi-modal information retrieval

## Dynamic RAG
* Real-time updating of knowledge bases to ensure the most current information
* Adaptive retrieval strategies based on query complexity and context

## Hierarchical RAG
* Multi-step retrieval processes that break complex queries into sub-queries
* Improved handling of long-form content and complex reasoning tasks

## Personalized RAG
* Tailoring retrieval based on user preferences, history, and context
* Integration with personal knowledge bases for more relevant responses

## RAG-Optimized Fine-Tuning
* Fine-tuning LLMs specifically for better performance with RAG systems
* Developing models that can more effectively utilize external knowledge

## Explainable RAG
* Improved transparency in the retrieval process
* Providing clear citations and sources for retrieved information
