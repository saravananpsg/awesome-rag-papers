# RAG: Latest trends
This repo is understand the latest trends of RAG and LLM
### Retrieval-Augmented Generation (RAG)

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
