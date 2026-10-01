# LLM Book Recommender
# Project Overview

This project uses Large Language Models (LLMs), vector embeddings, and sentiment analysis to recommend books based on natural language queries.

Users can:

- 🔎Search for books using semantic similarity
- 🏷️Filter books by category
- ☺️Filter books by emotional tone
- 🖥️Interact with recommendations through a Gradio web dashboard

# Technologies Used
- Python
- LangChain
- Hugging Face Transformers
- Vector Embeddings
- ChromaDB
- Gradio
- Pandas
- NLP

# Project Pipeline
## 1. Data Cleaning

Book metadata and descriptions were cleaned and standardized before processing.

## 2. Vector Search

Book descriptions were converted into document embeddings using transformer-based models.

Embeddings capture semantic meaning and allow similarity-based search rather than keyword matching.

Example:

Query:
"Books about the Roman Empire"
 
System:
Finds books whose embeddings are closest to the query embedding.

## 3. Text Classification

Zero-shot classification was used to assign categories to books.

Example:

"A heartwarming journey of love and friendship"
 
→ Fiction

This classification allows users to filter recommendations by category.

## 4. Sentiment Analysis

Book descriptions were classified into emotional categories:

- 😊Joy
- 🥹Sadness
- 😨Fear
- 😡Anger
- 🤢Disgust
- 😲Surprise
- 😐Neutral

Example:

"A heartwarming journey of love and friendship"
 
→ Joy

This enables mood-based book recommendations.

## 5. Gradio Dashboard

A Gradio application was developed to provide an interactive interface where users can:

- Enter a search query
- Filter by category
- Filter by emotion
- Receive book recommendations

# How It Works
## Embeddings

Traditional keyword search looks for exact matches.

This project uses transformer-based embeddings to represent text as vectors in semantic space.

Books discussing similar topics are located near one another within the vector database.

## Vector Database

Embeddings are stored in a vector database where similarity search retrieves the most relevant books.

## Future Improvements
- Hybrid recommendations
- User ratings and feedback
- Recommendation explanations
- Fine-tuned classification models
- Larger book datasets
