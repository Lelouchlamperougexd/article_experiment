# RAG Experiment with Wikipedia and ArXiv datasets using Hugging Face models
# For Google Colab

# Install required packages
!pip install datasets pandas numpy torch tqdm faiss-cpu sentence-transformers transformers openpyxl

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import os
import json
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. Loading and preparing data
# ==============================

print("Loading datasets...")

# Loading Wikipedia-Articles dataset
try:
    dataset_wiki = load_dataset("BrightData/Wikipedia-Articles", split="train")
    docs_wiki_df = pd.DataFrame(dataset_wiki)
    # If the dataset doesn't have an explicit identifier, create one by index
    if 'doc_id' not in docs_wiki_df.columns:
        docs_wiki_df['doc_id'] = "wiki_" + docs_wiki_df.index.astype(str)
    # Process document text (concatenating title and text)
    docs_wiki_df['full_text'] = docs_wiki_df['title'] + ". " + docs_wiki_df['cataloged_text'].fillna("")
    print(f"Loaded {len(docs_wiki_df)} documents from Wikipedia")
except Exception as e:
    print(f"Error loading Wikipedia-Articles: {e}")
    docs_wiki_df = pd.DataFrame()

# Loading ArXiv dataset
try:
    dataset_arxiv = load_dataset("arxiv_dataset", split="train[:1000]")  # Limiting for example
    docs_arxiv_df = pd.DataFrame(dataset_arxiv)
    # Create identifier
    if 'doc_id' not in docs_arxiv_df.columns:
        docs_arxiv_df['doc_id'] = "arxiv_" + docs_arxiv_df.index.astype(str)
    # Process document text
    if 'abstract' in docs_arxiv_df.columns and 'title' in docs_arxiv_df.columns:
        docs_arxiv_df['full_text'] = docs_arxiv_df['title'] + ". " + docs_arxiv_df['abstract'].fillna("")
    print(f"Loaded {len(docs_arxiv_df)} documents from ArXiv")
except Exception as e:
    print(f"Error loading ArXiv: {e}")
    # Create sample ArXiv data if failed to load the dataset
    docs_arxiv_df = pd.DataFrame({
        'doc_id': [f"arxiv_{i}" for i in range(10)],
        'title': [f"ArXiv Paper {i}" for i in range(10)],
        'abstract': [f"Abstract of paper {i} discussing scientific topics." for i in range(10)],
        'full_text': [f"ArXiv Paper {i}. Abstract of paper {i} discussing scientific topics." for i in range(10)]
    })

# Merge datasets (optional)
all_docs_df = pd.concat([docs_wiki_df, docs_arxiv_df], ignore_index=True)
print(f"Total documents: {len(all_docs_df)}")

# Create query set for experiment
queries = [
    {"query_id": "q1", "query_text": "History of Wikipedia", "dataset": "wiki"},
    {"query_id": "q2", "query_text": "Climate change", "dataset": "both"},
    {"query_id": "q3", "query_text": "Artificial intelligence and machine learning", "dataset": "both"},
    {"query_id": "q4", "query_text": "Quantum mechanics", "dataset": "arxiv"},
    {"query_id": "q5", "query_text": "Deep learning in computer vision", "dataset": "arxiv"},
]
queries_df = pd.DataFrame(queries)
print("Query set:")
print(queries_df)

# 2. Retrieval Components
# ======================

# Path to save indexes
INDEX_DIR = "./search_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# Function to create a simple inverted index without pyserini
def create_simple_index(documents_df):
    """Creates a simple inverted index for BM25-like search"""
    print("Creating simple BM25-like index...")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents_df['full_text'])
    
    # Create dictionaries for search
    vocabulary = vectorizer.vocabulary_
    idf = vectorizer.idf_
    doc_vectors = X
    
    return {
        'vectorizer': vectorizer,
        'vocabulary': vocabulary,
        'idf': idf,
        'doc_vectors': doc_vectors,
        'document_ids': documents_df['doc_id'].tolist()
    }

# Create indexes for search
# For BM25 we use a simple index instead of pyserini
bm25_index = create_simple_index(all_docs_df)

# Function to search using simple index
def simple_bm25_search(query, index, top_k=10):
    vectorizer = index['vectorizer']
    doc_vectors = index['doc_vectors']
    document_ids = index['document_ids']
    
    # Vectorize query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarity
    similarity_scores = query_vector.dot(doc_vectors.T).toarray()[0]
    
    # Sort results
    top_indices = similarity_scores.argsort()[-top_k:][::-1]
    results = [(document_ids[idx], similarity_scores[idx]) for idx in top_indices]
    
    return results

# Function for BM25 search
def bm25_search(query, top_k=10):
    # Use simple index instead of pyserini
    return simple_bm25_search(query, bm25_index, top_k)

# Dense search with SentenceTransformer and FAISS
print("Creating vector index...")
# Load pre-trained model
model_name = 'all-MiniLM-L6-v2'  # Lightweight model, works quickly
try:
    model = SentenceTransformer(model_name)
    
    # Create list of document texts for encoding
    doc_texts = all_docs_df['full_text'].tolist()
    
    # Encode documents into vector representation (using batches to save memory)
    batch_size = 32
    doc_embeddings = None
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Vectorizing documents"):
        batch_texts = doc_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        batch_embeddings_np = batch_embeddings.cpu().numpy()
        
        if doc_embeddings is None:
            doc_embeddings = batch_embeddings_np
        else:
            doc_embeddings = np.vstack((doc_embeddings, batch_embeddings_np))
    
    # Create FAISS index for vector search
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(doc_embeddings)
    index.add(doc_embeddings)
    
    # Save objects for search function
    dense_search_objects = {
        'model': model,
        'index': index,
        'doc_ids': all_docs_df['doc_id'].tolist()
    }
    
    def dense_search(query, top_k=10):
        # Encode query
        q_emb = dense_search_objects['model'].encode([query], convert_to_tensor=True)
        q_emb = q_emb.cpu().numpy()
        # Normalize for cosine similarity
        faiss.normalize_L2(q_emb)
        # Search
        scores, indices = dense_search_objects['index'].search(q_emb, k=top_k)
        # Format results list
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(dense_search_objects['doc_ids']):
                doc_id = dense_search_objects['doc_ids'][idx]
                results.append((doc_id, float(score)))
        return results
    
except Exception as e:
    print(f"Error creating vector index: {e}")
    # Create fallback if index creation failed
    def dense_search(query, top_k=10):
        print("Dense search failed to initialize. Returning empty results.")
        return []

# Hybrid search
def normalize_scores(results):
    """Normalizes scores to range [0, 1]"""
    if not results:
        return {}
    
    docs_scores = {doc_id: score for doc_id, score in results}
    scores = list(docs_scores.values())
    
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return {doc_id: 1.0 for doc_id in docs_scores}
    
    norm_scores = {
        doc_id: (score - min_score) / (max_score - min_score) 
        for doc_id, score in docs_scores.items()
    }
    return norm_scores

def hybrid_search(query, lambda_weight=0.5, top_k=10):
    """Combines results from BM25 and dense search"""
    # Get results from both search methods
    bm25_results = bm25_search(query, top_k=top_k*2)  # Get more to have something to combine
    dense_results = dense_search(query, top_k=top_k*2)
    
    # Normalize scores
    bm25_norm = normalize_scores(bm25_results)
    dense_norm = normalize_scores(dense_results)
    
    # Combine document IDs from both methods
    all_docs = set(bm25_norm.keys()).union(set(dense_norm.keys()))
    
    # Calculate hybrid scores
    hybrid_scores = {}
    for doc in all_docs:
        score_bm25 = bm25_norm.get(doc, 0)
        score_dense = dense_norm.get(doc, 0)
        hybrid_scores[doc] = lambda_weight * score_bm25 + (1 - lambda_weight) * score_dense
    
    # Sort by combined score
    hybrid_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return hybrid_results[:top_k]

# 3. Generative Component
# =======================

print("Loading generative model...")
try:
    # Use small FLAN-T5 model for generation
    model_name = "google/flan-t5-small"  # Small model, fast, works on CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Create pipeline for generation
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )
    
    def generate_answer(query, contexts, max_context_length=1500):
        """Generates answer based on query and contexts"""
        # Limit context length for model
        combined_context = " ".join(contexts)
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length]
        
        # Form prompt for generative model
        prompt = f"""
Answer the question based on the given context.

Context: {combined_context}

Question: {query}

Answer:"""
        
        # Generate answer
        output = generator(prompt, max_length=150, num_return_sequences=1)
        return output[0]['generated_text']
    
except Exception as e:
    print(f"Error loading generative model: {e}")
    # Create fallback if model loading failed
    def generate_answer(query, contexts, max_context_length=1500):
        return "Failed to load generative model. Here are the contexts that were found: " + " [...] ".join(contexts[:3])

# 4. Complete RAG Pipeline
# =====================

def rag_pipeline(query, search_method="hybrid", num_docs=3):
    """
    Complete RAG pipeline:
    1. Find relevant documents
    2. Extract contexts
    3. Generate answer
    
    Args:
        query: query text
        search_method: search method ('bm25', 'dense', or 'hybrid')
        num_docs: number of documents to use in context
    """
    # 1. Get search results based on chosen method
    if search_method == "bm25":
        search_results = bm25_search(query, top_k=num_docs)
    elif search_method == "dense":
        search_results = dense_search(query, top_k=num_docs)
    else:  # hybrid
        search_results = hybrid_search(query, top_k=num_docs)
    
    # 2. Extract document texts
    doc_ids = [doc_id for doc_id, _ in search_results]
    contexts = []
    for doc_id in doc_ids:
        # Find document in DataFrame
        doc_row = all_docs_df[all_docs_df['doc_id'] == doc_id]
        if not doc_row.empty:
            contexts.append(doc_row['full_text'].values[0])
    
    # 3. Generate answer
    answer = generate_answer(query, contexts)
    
    return {
        "query": query,
        "search_results": search_results,
        "contexts": contexts,
        "answer": answer
    }

# 5. Results Evaluation
# ===================

# Function to save results for subsequent evaluation
def save_results_for_evaluation(results, output_file="rag_evaluation_results.json"):
    """Saves results for subsequent expert evaluation"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_file}")
    
    # For Google Colab - also download the file
    try:
        from google.colab import files
        files.download(output_file)
    except:
        pass

# Run experiment with different search methods
def run_experiment():
    all_results = []
    
    methods = ["bm25", "dense", "hybrid"]
    
    # Process all queries
    for _, query_row in queries_df.iterrows():
        query_id = query_row['query_id']
        query_text = query_row['query_text']
        
        query_results = {
            "query_id": query_id,
            "query_text": query_text,
            "methods": {}
        }
        
        # Execute RAG with each search method
        for method in methods:
            print(f"Executing query {query_id} with method {method}...")
            try:
                result = rag_pipeline(query_text, search_method=method)
                query_results["methods"][method] = {
                    "search_results": [
                        {"doc_id": doc_id, "score": float(score)} 
                        for doc_id, score in result["search_results"]
                    ],
                    "answer": result["answer"]
                }
            except Exception as e:
                print(f"Error executing RAG with method {method}: {e}")
                query_results["methods"][method] = {
                    "search_results": [],
                    "answer": f"Error: {str(e)}"
                }
        
        all_results.append(query_results)
    
    # Save results
    save_results_for_evaluation(all_results)
    
    # Display brief information about results
    print("\nExperiment results:")
    for result in all_results:
        print(f"\nQuery: {result['query_text']}")
        for method, data in result["methods"].items():
            print(f"  Method: {method}")
            print(f"  Documents found: {len(data['search_results'])}")
            print(f"  First 100 characters of answer: {data['answer'][:100]}...")
    
    return all_results

# 6. Create template for expert evaluation
# ======================================

def create_evaluation_template(results, output_file="expert_evaluation_template.xlsx"):
    """Creates Excel template for expert evaluation"""
    try:
        evaluation_data = []
        
        for result in results:
            query_id = result["query_id"]
            query_text = result["query_text"]
            
            for method, data in result["methods"].items():
                # Add rows for answer evaluation
                evaluation_data.append({
                    "query_id": query_id,
                    "query_text": query_text,
                    "method": method,
                    "doc_id": "N/A",  # For answer evaluation
                    "title": "ANSWER",
                    "text": data["answer"],
                    "relevance_score": "",  # For expert to fill in
                    "comments": ""  # For expert to fill in
                })
                
                # Add rows for found document evaluation
                for i, doc_info in enumerate(data["search_results"]):
                    doc_id = doc_info["doc_id"]
                    doc_row = all_docs_df[all_docs_df['doc_id'] == doc_id]
                    
                    if not doc_row.empty:
                        title = doc_row['title'].values[0] if 'title' in doc_row.columns else "No title"
                        text = doc_row['full_text'].values[0][:500] + "..."  # Trim for readability
                        
                        evaluation_data.append({
                            "query_id": query_id,
                            "query_text": query_text,
                            "method": method,
                            "doc_id": doc_id,
                            "title": title,
                            "text": text,
                            "relevance_score": "",  # For expert to fill in
                            "comments": ""  # For expert to fill in
                        })
        
        # Create DataFrame and save to Excel
        eval_df = pd.DataFrame(evaluation_data)
        eval_df.to_excel(output_file, index=False)
        print(f"Expert evaluation template created: {output_file}")
        
        # For Google Colab - also download the file
        try:
            from google.colab import files
            files.download(output_file)
        except:
            pass
    
    except Exception as e:
        print(f"Error creating evaluation template: {e}")

# 7. Run the experiment
# ====================

# Run the experiment
print("Starting RAG experiment...")
results = run_experiment()
create_evaluation_template(results)
print("Experiment completed!")

# 8. Custom test function
# =====================

def test_rag_with_custom_query(query, search_method="hybrid"):
    """Test RAG with a custom query to see how it performs"""
    print(f"\nTesting with custom query: '{query}'")
    print(f"Using search method: {search_method}")
    
    result = rag_pipeline(query, search_method=search_method)
    
    print("\nSearch results:")
    for i, (doc_id, score) in enumerate(result["search_results"]):
        doc_row = all_docs_df[all_docs_df['doc_id'] == doc_id]
        if not doc_row.empty:
            title = doc_row['title'].values[0] if 'title' in doc_row.columns else "No title"
            print(f"{i+1}. {title} (score: {score:.4f})")
    
    print("\nGenerated answer:")
    print(result["answer"])
    
    return result

# Example custom query test
test_rag_with_custom_query("What are the benefits of renewable energy?")