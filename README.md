# YouTube RAG Chatbot

## Overview

The YouTube RAG Chatbot is a Python-based application that allows users to ask questions about YouTube videos. It extracts video transcripts, processes the text, retrieves relevant information, and uses a Large Language Model (LLM) to generate answers based on the video content.

The project uses **Retrieval-Augmented Generation (RAG)** to generate context-based answers rather than relying only on the LLM's general knowledge.

## Features

- **YouTube Transcript Extraction:** Extracts available English or Hindi transcripts.
- **Hindi-to-English Translation:** Translates Hindi transcripts into English using Helsinki-NLP.
- **Text Chunking:** Splits transcripts into smaller chunks for efficient retrieval.
- **Text Embeddings:** Converts transcript chunks into numerical vectors.
- **Vector Search:** Uses FAISS to retrieve relevant transcript chunks.
- **Local LLM:** Uses Qwen2.5-0.5B-Instruct to generate answers locally.
- **Context-Based Responses:** Uses retrieved transcript chunks to answer questions.
- **No Paid Inference API Required:** Runs the LLM locally using `llama.cpp`.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YouTube Transcript API | Extracts YouTube transcripts |
| Helsinki-NLP/opus-mt-hi-en | Translates Hindi into English |
| RecursiveCharacterTextSplitter | Splits transcripts into chunks |
| all-MiniLM-L6-v2 | Generates text embeddings |
| FAISS | Stores and searches vectors |
| Qwen2.5-0.5B-Instruct-GGUF | Generates answers |
| llama.cpp | Runs the local LLM |
| LangChain | Connects the RAG pipeline components |

## System Architecture

1. **Video Input:** The user provides a YouTube video ID.
2. **Transcript Extraction:** The application retrieves an available English or Hindi transcript, preferring English.
3. **Translation:** Hindi transcripts are translated into English.
4. **Text Chunking:** The transcript is divided into smaller, overlapping chunks.
5. **Embedding Generation:** Each chunk is converted into a vector using the embedding model.
6. **Vector Store Creation:** The vectors and corresponding text are stored in FAISS.
7. **Question Processing:** The user's question is converted into an embedding.
8. **Information Retrieval:** FAISS retrieves the most relevant transcript chunks.
9. **Answer Generation:** The retrieved context and question are passed to the local Qwen model.
10. **Response:** The generated answer is displayed to the user.

## Models Used

### 1. Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

Converts text into numerical vectors representing semantic meaning. FAISS uses these vectors to find transcript chunks relevant to the user's question.

### 2. Translation Model

**Model:** `Helsinki-NLP/opus-mt-hi-en`

Translates Hindi transcripts into English before they enter the retrieval pipeline.

### 3. Large Language Model (LLM)

**Model:** `Qwen2.5-0.5B-Instruct-GGUF`

This quantized language model runs locally using `llama.cpp`. It receives the retrieved transcript chunks and generates an answer to the user's question.