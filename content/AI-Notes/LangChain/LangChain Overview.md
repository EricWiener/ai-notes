---
tags: [flashcards]
source:
summary:
---
[Good 13 minute YouTube Video](https://youtu.be/aywZrzNaKjs)
### Main Concepts of LangChain
**Components:**
- LLM wrappers (access models from OpenAI and Hugging Face)
- Prompt templates (hard code text as input to the LLMs)
- Indexes for relevant information retrieval (extract relevant info for LLMs)

**Chains:**
Combine components together to build an application.

**Agents:**
Allow LLMs to interact with external APIs. LangChain is agentic (==the agent can take actions and not only provide answers to questions==).
<!--SR:!2024-01-21,85,310-->

### Question/Answering Over Documents
You load your documents via a `document_loader` and then split up the documents via a `text_splitter`. You then embed your split up documents using an `embedding` and store these embeddings in a `VectorStore`.
<!--SR:!2023-08-18,4,270-->

Once you ask your question, you then:
- Send your question to a LLM.
- Embed your question via the `embedding` and do a similarity search in your `VectorStore` to find the relevant document embeddings. You then pass the relevant info to your LLM.

The LLM then has the question and relevant info it needs.

### VectorStores
- PineCone: a vector store
- ChromeDB: another vector store option.