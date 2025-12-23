# E-Commerce Search Information Retrieval System

## Overview
This solution implements a robust Information Retrieval (IR) system for Persian e-commerce search. It is designed to maximize **Precision@10** by using a two-stage Cascade Architecture: **Retrieval (Recall)** followed by **Learning-to-Rank (Precision)**.

## Approach & Methodology
The solution is inspired by the paper *"Exploring Query Understanding for Amazon Product Search"*, adapting its core concepts to a self-contained environment without external APIs.

### 1. Data Preprocessing (Normalization)
- **Persian Text Normalization:** Unified characters (Arabic/Persian 'y' and 'k'), removed Zero-Width Non-Joiners, and cleaned punctuation to ensure the query grammar matches the product catalog grammar.

### 2. Query Understanding (QU)
Instead of simple keyword matching, we built a Semantic Understanding layer:
- **Knowledge Base:** Extracted structured entities (Brands, Categories, Product Types) from the product catalog.
- **Intent Dictionary:** Mined `train_query_product_pairs` to learn probabilistic mappings (e.g., "گلکسی" implies `Brand: Samsung`).
- **Query Parser:** A rule-based NER system that tags query tokens with attributes (Brand, Category).

### 3. Retrieval (Recall Phase)
- **Algorithm:** BM25 (Best Matching 25).
- **Implementation:** A custom Inverted Index built from scratch to efficiently retrieve the top 100 candidates from 19,000+ products.
- **Goal:** To narrow down the search space rapidly.

### 4. Ranking (Precision Phase)
- **Model:** LightGBM (Gradient Boosting Machine) trained with a Pointwise Learning-to-Rank objective.
- **Features:**
    - **Semantic Features:** Brand Match, Category Match (Derived from QU).
    - **Lexical Features:** BM25 Score, Jaccard Similarity, Fuzzy Matching.
    - **Query Features:** Query Length, Specificity.
- **Training:** Used "Hard Negative Mining" (training on retrieved items that were *not* the target) to teach the model to distinguish subtle differences.

## File Structure
- `text_processor.py`: Core text cleaning logic.
- `knowledge_base.py` & `intent_dictionary.py`: Knowledge extraction.
- `retriever.py`: BM25 Search Engine.
- `feature_extractor.py`: Generates ML features from query-product pairs.
- `ranker.py`: LightGBM model wrapper.
- `inference_engine.py`: Runs the full pipeline on test data.
- `main_step11.py`: Main execution script.

## Installation & Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt