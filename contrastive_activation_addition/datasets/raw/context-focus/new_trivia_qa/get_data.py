from datasets import load_dataset
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import wikipedia
import re
import numpy as np
from tqdm import tqdm
import json

# Download required NLTK data
nltk.download('punkt')

def split_into_paragraphs(text, min_length=100):
    """Split text into paragraphs with minimum length."""
    if not text:
        return []
    
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    # Filter out short paragraphs
    return [p for p in paragraphs if len(p) >= min_length]

def create_bm25_index(paragraphs):
    """Create BM25 index from paragraphs."""
    # Tokenize paragraphs
    tokenized_paragraphs = [word_tokenize(paragraph.lower()) for paragraph in paragraphs]
    # Create BM25 index
    return BM25Okapi(tokenized_paragraphs), tokenized_paragraphs

def retrieve_top_paragraph(bm25, tokenized_paragraphs, query, paragraphs, k=1):
    """Retrieve top-k paragraphs for a given query."""
    # Tokenize query
    tokenized_query = word_tokenize(query.lower())
    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)
    # Get top-k indices
    top_k_indices = np.argsort(scores)[-k:][::-1]
    
    return [(paragraphs[idx], scores[idx]) for idx in top_k_indices]

def process_trivia_qa():
    """Process a subset of TriviaQA dataset and create question-context-answer triplets."""
    # Load TriviaQA dataset
    with open("triviaqa_data.json",'r') as f:
        dataset = json.load(f)

    triplets = []
    
    for item in tqdm(dataset):
        question = item['question']
        answer = item['answer']
        contexts = item['wiki_contexts']
        content = "\n".join(contexts)

        # Split into paragraphs
        paragraphs = split_into_paragraphs(content)
        if not paragraphs:
            continue
        
        # Create BM25 index
        bm25, tokenized_paragraphs = create_bm25_index(paragraphs)
        
        # Retrieve top paragraph
        top_results = retrieve_top_paragraph(bm25, tokenized_paragraphs, question, paragraphs, k=1)

        if top_results:
            context, score = top_results[0]
            print(context,"\n\n",score)

            triplets.append({
                'question': question,
                'context': context,
                'answer': answer,
                'score': score
            })
    
    return triplets

def main():
    # Process dataset and create triplets
    print("Processing TriviaQA dataset...")
    triplets = process_trivia_qa()
    
    # Save results
    print(f"Generated {len(triplets)} question-context-answer triplets")
    
    with open("triviaqa_retrieved.json",'w') as f:
        json.dump(triplets,f)

if __name__ == "__main__":
    main()