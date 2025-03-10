# Install dependencies
!pip install datasets pandas numpy torch tqdm faiss-cpu sentence-transformers transformers rank_bm25 llama-index kaggle

# Import required libraries
import os
import json
import pandas as pd
import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from rank_bm25 import BM25Okapi
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set up Kaggle API and download dataset
print("Downloading ArXiv dataset from Kaggle...")
os.environ["KAGGLE_CONFIG_DIR"] = "/content"
!kaggle datasets download -d cornell-university/arxiv --unzip

# Load ArXiv dataset
print("Loading ArXiv dataset...")
FILE_PATH = "arxiv-metadata-oai-snapshot.json"
with open(FILE_PATH, "r") as f:
    arxiv_data = [json.loads(line) for line in f]

docs_arxiv_df = pd.DataFrame(arxiv_data)
docs_arxiv_df = docs_arxiv_df[['id', 'title', 'abstract', 'categories', 'update_date']]
docs_arxiv_df['full_text'] = docs_arxiv_df['title'] + ". " + docs_arxiv_df['abstract'].fillna("")
docs_arxiv_df['update_date'] = pd.to_datetime(docs_arxiv_df['update_date'], errors='coerce')
docs_arxiv_df = docs_arxiv_df.dropna(subset=['update_date'])

# Filter papers (old <2020, new >=2021)
print("Filtering dataset...")
old_docs_df = docs_arxiv_df[docs_arxiv_df['update_date'] < '2020-01-01'].sample(n=5000, random_state=42)
new_docs_df = docs_arxiv_df[docs_arxiv_df['update_date'] >= '2021-01-01'].sample(n=5000, random_state=42)

# Create FAISS index
print("Creating FAISS index...")
model = SentenceTransformer('all-MiniLM-L6-v2')
old_embeddings = model.encode(old_docs_df['full_text'].tolist(), convert_to_tensor=True).cpu().numpy()
faiss.normalize_L2(old_embeddings)
index = faiss.IndexFlatIP(old_embeddings.shape[1])
index.add(old_embeddings)

# Create BM25 Index
print("Creating BM25 index...")
tokenized_corpus = [doc.split() for doc in old_docs_df['full_text'].tolist()]
bm25 = BM25Okapi(tokenized_corpus)

# Initialize LLM for generation - используем общедоступную модель
print("Loading language model for generation...")
llm_model_name = "facebook/opt-1.3b"  # Меньшая и общедоступная модель
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
generator = pipeline(
    "text-generation",
    model=llm_model_name,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
    max_length=512
)

def generate_text(prompt, max_length=512):
    """Generate text using the language model pipeline."""
    response = generator(
        prompt,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        num_return_sequences=1
    )
    # Pipeline возвращает список словарей
    return response[0]['generated_text'][len(prompt):]  # Возвращаем только новый текст

# Define Novelty Evaluation Function
def evaluate_novelty(new_text, top_k=5, alpha=0.5):
    """Evaluate novelty using hybrid search and keyword uniqueness."""
    # FAISS retrieval
    new_embedding = model.encode([new_text], convert_to_tensor=True).cpu().numpy()
    faiss.normalize_L2(new_embedding)
    scores, indices = index.search(new_embedding, k=top_k)
    avg_similarity = np.mean(scores)
    
    # Get most similar documents
    similar_docs = [old_docs_df.iloc[idx]['full_text'] for idx in indices[0]]
    similar_titles = [old_docs_df.iloc[idx]['title'] for idx in indices[0]]

    # BM25 retrieval
    bm25_scores = bm25.get_scores(new_text.split())
    top_bm25_indices = np.argsort(bm25_scores)[-top_k:]
    avg_bm25_score = np.mean([bm25_scores[i] for i in top_bm25_indices])
    
    # Get BM25 most similar documents
    bm25_docs = [old_docs_df.iloc[idx]['full_text'] for idx in top_bm25_indices]
    bm25_titles = [old_docs_df.iloc[idx]['title'] for idx in top_bm25_indices]

    # Keyword uniqueness
    words = Counter(new_text.split())
    unique_score = sum(1 for w in words if words[w] == 1) / len(words)

    # Combined novelty score
    novelty_score = (1 - avg_similarity) * alpha + unique_score * (1 - alpha)
    
    return novelty_score, similar_docs, similar_titles, bm25_docs, bm25_titles

# Function to implement full RAG system
def rag_novelty_analysis(paper_text, paper_title=""):
    """
    Full RAG implementation for paper novelty analysis:
    1. Retrieval: Find similar papers
    2. Augmentation: Combine retrieval results with paper
    3. Generation: Generate insights about the paper's novelty
    """
    # Step 1: Retrieval - get similar papers and novelty score
    novelty_score, vector_docs, vector_titles, keyword_docs, keyword_titles = evaluate_novelty(paper_text)
    
    # Step 2: Augmentation - create prompt with retrieved information
    prompt = f"""Paper Title: {paper_title}
Paper Abstract: {paper_text[:300]}...

This paper has a novelty score of {novelty_score:.4f} (higher means more novel).

Most similar papers by semantic search:
1. {vector_titles[0]}
2. {vector_titles[1]}
3. {vector_titles[2]}

Most similar papers by keyword matching:
1. {keyword_titles[0]}
2. {keyword_titles[1]}
3. {keyword_titles[2]}

Analyze the novelty of this paper based on the above information. Discuss:
1. How novel is this research?
2. What makes it innovative?
3. How does it relate to existing literature?
4. How could its novelty be improved?

Analysis:
"""
    
    # Step 3: Generation - generate novelty analysis
    analysis = generate_text(prompt)
    
    return {
        "novelty_score": novelty_score,
        "similar_papers_semantic": vector_titles[:3],
        "similar_papers_keyword": keyword_titles[:3],
        "analysis": analysis
    }

# Evaluate novelty for new papers with RAG
print("Evaluating novelty of new papers with RAG...")
rag_results = []
# Use fewer papers for demonstration due to generation overhead
for _, row in tqdm(new_docs_df.head(5).iterrows(), total=5):
    result = rag_novelty_analysis(row['full_text'], row['title'])
    result["doc_id"] = row['id']
    result["title"] = row['title']
    rag_results.append(result)

# Save Results
rag_df = pd.DataFrame(rag_results).sort_values(by='novelty_score', ascending=False)
rag_df.to_csv("rag_novelty_results.csv", index=False)
print("Saved RAG novelty results to rag_novelty_results.csv")

# Display results
print("Top most novel papers with RAG analysis:")
for i, row in rag_df.head(3).iterrows():
    print(f"\n{'-'*80}")
    print(f"Title: {row['title']}")
    print(f"Novelty Score: {row['novelty_score']:.4f}")
    print(f"\nNovelty Analysis:\n{row['analysis']}")

# User interface function to analyze any paper
def analyze_user_paper(title, abstract):
    """Analyze any user-provided paper with the RAG system."""
    full_text = f"{title}. {abstract}"
    result = rag_novelty_analysis(full_text, title)
    
    print(f"\n{'-'*80}")
    print(f"Title: {title}")
    print(f"Novelty Score: {result['novelty_score']:.4f}")
    print("\nSimilar papers (semantic):")
    for i, title in enumerate(result['similar_papers_semantic']):
        print(f"{i+1}. {title}")
    print("\nSimilar papers (keyword):")
    for i, title in enumerate(result['similar_papers_keyword']):
        print(f"{i+1}. {title}")
    print(f"\nNovelty Analysis:\n{result['analysis']}")

# Alternative version: without running a heavy model for demonstration
def simplified_demo_analysis(paper_text, paper_title=""):
    """
    Demonstration version without running a generative model.
    This allows us to see how RAG works even if the model is unavailable.
    """
    # Step 1: Retrieval - get similar papers and novelty score
    novelty_score, vector_docs, vector_titles, keyword_docs, keyword_titles = evaluate_novelty(paper_text)
    
    # In the real RAG system, a generative model would be called here
    analysis = (f"Analysis of the paper '{paper_title}':\n"
                f"This work has a novelty score of {novelty_score:.4f}.\n"
                f"The most similar papers indicate that this research falls within an active area.\n"
                f"To increase novelty, it is recommended to focus on aspects that are missing in similar works.")
    
    return {
        "novelty_score": novelty_score,
        "similar_papers_semantic": vector_titles[:3],
        "similar_papers_keyword": keyword_titles[:3],
        "analysis": analysis
    }

# Example usage with a cryptocurrency-related paper:
user_title = "Cryptocurrencies as a tool for money laundering"
user_abstract = """The purpose of this study is to describe the opportunities and limitations of cryptocurrencies as a tool for money laundering through six currently available "open doors" (exchange mechanisms). The authors link the regulatory dialectic paradigm to know your customer and anti-money laundering evasion techniques, highlight six tactics to launder funds with virtual assets and investigate potential law enforcement and regulatory alternatives used to reduce the incidence of money laundering with digital coins."""

print("\nAnalyzing user-provided paper...")
analyze_user_paper(user_title, user_abstract)

# If there are issues with the model, the simplified_demo_analysis function can be used:
print("\nDemonstration analysis without using a generative model:")
result = simplified_demo_analysis(user_title + ". " + user_abstract, user_title)
print(f"Novelty Score: {result['novelty_score']:.4f}")
print(f"Analysis: {result['analysis']}")
