# Introduction to Retrieval-Augmented Generation (RAG)

> A research-oriented guide to the concepts, components, design choices, and evaluation of Retrieval-Augmented Generation. For the curated bibliography, see the [Awesome RAG Papers README](README.md).

## 1. What is RAG?

Retrieval-Augmented Generation (RAG) combines **retrieval from an external knowledge source** with **language-model generation**. Instead of asking a model to answer exclusively from information learned during training, a RAG system can retrieve potentially relevant evidence at inference time and use it to produce an answer.

A typical RAG workflow has an **offline knowledge-preparation path** and an **online question-answering path**. The two meet at retrieval:

```text
OFFLINE KNOWLEDGE PREPARATION                ONLINE QUESTION ANSWERING
-----------------------------                --------------------------
Documents / databases / graphs                 User question
              |                                      |
              v                                      v
     Parse and normalize                     Optional rewrite / expand
              |                                      |
              v                                      |
    Chunk or structure data                           |
              |                                      |
              v                                      v
     Embed / build index ----------------> Retrieve candidates
                                                     |
                                                     v
                                        Rerank / filter / compress
                                                     |
                                                     v
                                         Assemble evidence context
                                                     |
                                                     v
                                               Language model
                                                     |
                                                     v
                                            Answer + attribution
                                                     |
                                                     v
                                      Evaluate answer and retrieval
```

The index is typically prepared or refreshed independently of each user query. Some systems retrieve iteratively, while adaptive or agentic systems may decide **when to retrieve again**, query additional sources, or call tools during answer generation.

Retrieval **can improve access to current or domain-specific information**, but it does not automatically guarantee correctness. Success depends on document coverage, chunking, retrieval quality, evidence selection, model behavior, and the evaluation protocol.

## 2. Core technical decisions

| Component | Questions researchers study | Related bibliography categories |
| --- | --- | --- |
| Knowledge representation | How should documents be parsed, chunked, embedded, and indexed? | Chunking/Indexing |
| Query handling | When should a question be rewritten, expanded, clarified, or decomposed? | Query Enhancement |
| Candidate retrieval | Which passages, nodes, paths, or documents should be retrieved? | Query Enhancement; Multi-Hop & Knowledge Graphs |
| Context construction | Which evidence should be retained, reranked, filtered, or compressed? | Context Optimization; Long Context |
| Retrieval orchestration | Should retrieval happen once, repeatedly, or only when necessary? | Iterative Retrieval & Reasoning; Adaptive Retrieval |
| Planning and tools | How should a system choose actions, tools, and multi-step evidence-gathering strategies? | Tools & Planning; Advanced Architectures |
| Grounded generation | How should a model handle conflicting evidence and parametric knowledge? | Knowledge Conflicts & Factuality; Preference Alignment |
| Quality and safety | How do we measure factuality, robustness, retrieval quality, security, and privacy? | Evaluation & Analysis; Security & Privacy |

### Retrieval granularity and evidence types

RAG systems can retrieve evidence at several levels: **document-level** (entire reports or files), **passage-level** (chunks or sentences), **graph-level** (entities, relations, paths, or subgraphs), and **multimodal** (images, tables, audio, or combinations). The unit of retrieval affects indexing, recall, attribution, latency, and context cost. These are complementary choices rather than mutually exclusive system types.

### From naive to advanced and agentic RAG

| Pattern | Typical behavior | Main research challenge |
| --- | --- | --- |
| Naive RAG | Retrieve once, concatenate evidence, generate an answer | Relevant retrieval and grounded generation |
| Advanced RAG | Add query optimization, reranking, compression, self-critique, or specialized indexing | Choosing and integrating useful improvements |
| Agentic RAG | Plan evidence-gathering steps, invoke tools, or choose retrieval actions dynamically | Control policies, reliable tool use, and bounded cost |

These are broad architectural patterns, **not additional top-level bibliography categories**. In particular, iterative retrieval, adaptive retrieval, multi-hop reasoning, and tools/planning remain separate categories in the taxonomy below.

## 3. Research taxonomy: 5 parent groups, 15 categories

The bibliography **retains the original 15 technical categories**, organized under five navigational groups. The parent groups improve discoverability; they **do not replace** the underlying distinctions. Papers may relate to more than one category, while their original placement remains traceable in the migration inventory.

### I. Foundations & Architecture

1. **[Naive RAG](README.md#naive-rag)** — baseline retrieval-then-generation designs, foundational methods, and early empirical studies.
2. **[Advanced Architectures](README.md#advanced-architectures)** — modular, self-reflective, multi-component, and novel RAG architectures that go beyond the baseline pipeline.

### II. Retrieval & Context Engineering

3. **[Chunking/Indexing](README.md#chunkingindexing)** — document segmentation, representations, index structures, and retrieval granularity.
4. **[Query Enhancement](README.md#query-enhancement)** — query rewriting, expansion, intent interpretation, decomposition, and query-informed retrieval optimization.
5. **[Context Optimization](README.md#context-optimization)** — passage selection, reranking, noise filtering, compression, and evidence packing.
6. **[Long Context](README.md#long-context)** — interactions between long-context language models and retrieval, including context utilization and efficiency.

### III. Reasoning & Agentic RAG

7. **[Multi-Hop & Knowledge Graphs](README.md#multi-hop--knowledge-graphs)** — evidence chains across documents or structured graphs, graph retrieval, and compositional reasoning.
8. **[Iterative Retrieval & Reasoning](README.md#iterative-retrieval--reasoning)** — repeated retrieve–reason–refine loops, verification, and incremental evidence gathering.
9. **[Adaptive Retrieval](README.md#adaptive-retrieval)** — policies deciding **whether, when, or how much** to retrieve based on uncertainty, task requirements, or model state.
10. **[Tools & Planning](README.md#tools--planning)** — tool orchestration, action selection, search planning, and agentic evidence acquisition.

These four categories overlap in practice but describe **different research questions**. A system can be adaptive without performing multi-hop graph reasoning; it can perform iterative retrieval without autonomous tool planning.

### IV. Training & Trustworthiness

11. **[Preference Alignment](README.md#preference-alignment)** — training or alignment methods that influence how retrievers and generators use external evidence.
12. **[Knowledge Conflicts & Factuality](README.md#knowledge-conflicts--factuality)** — hallucination mitigation, faithfulness, uncertainty, and conflicts between **retrieved context and a model's parametric knowledge**. This is distinct from *evaluation*, which measures outcomes.
13. **[Security & Privacy](README.md#security--privacy)** — prompt injection, poisoning, privacy leakage, provenance, and secure retrieval practices.

### V. Applications & Evaluation

14. **[Specialized Domains & Modalities](README.md#specialized-domains--modalities)** — domain-specific and multimodal RAG, including text, images, tables, graphs, and specialized corpora.
15. **[Evaluation & Analysis](README.md#evaluation--analysis)** — benchmarks, metrics, diagnostics, datasets, and controlled comparisons of retrieval and generation.

## 4. How to assess a RAG paper

A useful reading framework is to record: **(1) the research problem; (2) the baseline; (3) what is retrieved; (4) retrieval/control decisions; (5) training or inference costs; (6) benchmark and evaluation protocol; (7) ablations and failure cases; and (8) available implementations.**

Be careful with results that compare systems using different retrieval corpora, embedding models, answer generators, context budgets, or evaluation methods: apparent improvements may come from the experimental setup rather than the proposed technique.

For systematic evaluation, distinguish **retrieval quality** (e.g., recall and ranking effectiveness), **answer quality** (correctness, faithfulness, and attribution), and **operational properties** (latency, token cost, robustness, and security). No single score captures all three.

### Evaluation dimensions and metrics

| Dimension | Example measurements | Interpretation |
| --- | --- | --- |
| Retrieval coverage | Recall@k, hit rate | Whether relevant evidence is present among retrieved candidates |
| Ranking | MRR, nDCG@k | Whether relevant evidence is ranked highly |
| Grounded generation | Answer correctness, faithfulness, citation attribution | Whether claims are correct and supported by retrieved sources |
| Control behavior | Retrieval decisions, tool-call success, multi-step task completion | Whether an adaptive or agentic policy selects useful actions |
| Operational performance | End-to-end latency (including P95), token usage, cost per query | Whether the system can meet deployment budgets |
| Trustworthiness | Robustness to noise, prompt injection, poisoned documents, privacy leakage | Whether the system behaves reliably under adverse conditions |

Metrics must be defined with respect to the evaluation data. For example, **Recall@k** requires a definition of relevant passages, while **faithfulness** measures support from provided evidence and is not interchangeable with world-factual correctness. Use identical corpora, retrievers, generators, and token budgets for controlled comparisons, or document those differences explicitly.

## 5. Suggested reading path

New to RAG? Start with **Naive RAG**, then study **Chunking/Indexing**, **Query Enhancement**, and **Context Optimization**. Next compare **Long Context** against retrieval-heavy approaches, then explore **Multi-Hop & Knowledge Graphs**, **Iterative Retrieval**, **Adaptive Retrieval**, and **Tools & Planning**. Finish with **Knowledge Conflicts & Factuality**, **Security & Privacy**, and **Evaluation & Analysis** to understand limitations and responsible deployment.

---

*Scope: this is an orientation guide, not a claim that all listed papers or code repositories have been independently verified.*
