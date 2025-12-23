# import math
# from collections import defaultdict, Counter
# from text_processor import PersianPreprocessor

# class BM25Retriever:
#     def __init__(self, k1=1.5, b=0.75):
#         """
#         k1: Controls term frequency saturation (typically 1.2 to 2.0).
#         b: Controls length normalization (0.75 is standard).
#         """
#         self.k1 = k1
#         self.b = b
#         self.preprocessor = PersianPreprocessor()
        
#         # Index structures
#         self.inverted_index = defaultdict(list) # {term: [doc_indices]}
#         self.doc_lengths = [] # [length of doc 0, length of doc 1, ...]
#         self.avg_dl = 0
#         self.doc_freqs = defaultdict(int) # {term: count of docs containing term}
#         self.corpus_size = 0
#         self.idf = {}
#         self.doc_ids = [] # Map internal index to real Product ID

#     def fit(self, products_df):
#         """
#         Builds the inverted index from the product catalog.
#         """
#         print("Indexing Corpus...")
#         self.corpus_size = len(products_df)
#         self.doc_ids = products_df['p_id'].tolist()
        
#         # We index the 'search_text' column created in Step 1
#         corpus_text = products_df['search_text'].tolist()
        
#         total_length = 0
        
#         for idx, text in enumerate(corpus_text):
#             # Tokenize
#             tokens = self.preprocessor.tokenize(text)
#             length = len(tokens)
#             self.doc_lengths.append(length)
#             total_length += length
            
#             # Update Term Frequencies and Inverted Index
#             # We use a set to count doc_freq only once per doc
#             unique_tokens = set(tokens)
            
#             # Count term frequencies for this document
#             term_counts = Counter(tokens)
            
#             for token in unique_tokens:
#                 self.doc_freqs[token] += 1
#                 # Store (doc_index, term_frequency_in_this_doc)
#                 self.inverted_index[token].append((idx, term_counts[token]))
                
#         self.avg_dl = total_length / self.corpus_size
#         self.calc_idf()
#         print(f"Index Built. Corpus Size: {self.corpus_size}, Avg Doc Length: {self.avg_dl:.2f}")

#     def calc_idf(self):
#         """
#         Precomputes IDF for all terms in the vocabulary.
#         Formula: IDF(q) = log( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )
#         """
#         for term, freq in self.doc_freqs.items():
#             # Standard Lucene/BM25 IDF formula
#             self.idf[term] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

#     def retrieve(self, query, top_k=100):
#         """
#         Retrieves top_k relevant documents for a query.
#         """
#         tokens = self.preprocessor.tokenize(query)
        
#         # Score accumulator: {doc_index: score}
#         scores = defaultdict(float)
        
#         for token in tokens:
#             if token not in self.inverted_index:
#                 continue
                
#             idf = self.idf[token]
            
#             # Iterate only over documents that contain this token
#             for doc_idx, freq in self.inverted_index[token]:
#                 doc_len = self.doc_lengths[doc_idx]
                
#                 # BM25 Scoring Function
#                 numerator = freq * (self.k1 + 1)
#                 denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
                
#                 scores[doc_idx] += idf * (numerator / denominator)
        
#         # Sort by score descending
#         sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
#         # Map back to Product IDs
#         results = []
#         for doc_idx, score in sorted_scores:
#             results.append({
#                 'p_id': self.doc_ids[doc_idx],
#                 'bm25_score': score
#             })
            
#         return results


import math
from collections import defaultdict, Counter
from text_processor import PersianPreprocessor

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.preprocessor = PersianPreprocessor()
        self.inverted_index = defaultdict(list)
        self.doc_lengths = []
        self.avg_dl = 0
        self.doc_freqs = defaultdict(int)
        self.corpus_size = 0
        self.idf = {}
        self.doc_ids = [] 

    def fit(self, products_df):
        print("Indexing Corpus...")
        self.corpus_size = len(products_df)
        
        # Force string IDs here too
        self.doc_ids = products_df['p_id'].astype(str).tolist()
        
        # Note: We use the ENRICHED search_text from DataLoader
        corpus_text = products_df['search_text'].tolist()
        total_length = 0
        
        for idx, text in enumerate(corpus_text):
            tokens = self.preprocessor.tokenize(text)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length
            
            unique_tokens = set(tokens)
            term_counts = Counter(tokens)
            
            for token in unique_tokens:
                self.doc_freqs[token] += 1
                self.inverted_index[token].append((idx, term_counts[token]))
                
        self.avg_dl = total_length / self.corpus_size
        self.calc_idf()
        print(f"Index Built. Size: {self.corpus_size}")

    def calc_idf(self):
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def retrieve(self, query, top_k=100):
        # We normalize query to handle Finglish/English mapping
        tokens = self.preprocessor.tokenize(query)
        scores = defaultdict(float)
        
        for token in tokens:
            if token not in self.inverted_index:
                continue
            idf = self.idf[token]
            for doc_idx, freq in self.inverted_index[token]:
                doc_len = self.doc_lengths[doc_idx]
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_dl))
                scores[doc_idx] += idf * (num / den)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{'p_id': self.doc_ids[i], 'bm25_score': s} for i, s in sorted_scores]