from datasets import load_dataset
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
import wikipedia
import re
import numpy as np
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')

def clean_wiki_text(text):
    """Clean Wikipedia text by removing special characters and extra whitespace."""
    text = re.sub(r'\[.*?\]', '', text)  # Remove [edit] and similar tags
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    return text.strip()

def get_wikipedia_content(url):
    """Extract content from Wikipedia URL."""
    try:
        # Extract page title from URL
        title = url.split('/')[-1]
        # Get Wikipedia page content
        page = wikipedia.page(title, auto_suggest=False)
        # Clean and return the content
        return clean_wiki_text(page.content)
    except Exception as e:
        print(f"Error retrieving Wikipedia content for {url}: {str(e)}")
        return None

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

def process_trivia_qa(num_samples=1000, split="test"):
    """Process a subset of TriviaQA dataset and create question-context-answer triplets."""
    # Load TriviaQA dataset with streaming enabled
    dataset = load_dataset("trivia_qa", "rc", split=split, streaming=True)
    
    # Convert streaming dataset to an iterator and take only num_samples
    dataset_iter = iter(dataset)
    
    triplets = []
    processed_count = 0
    
    with tqdm(total=num_samples) as pbar:
        while processed_count < num_samples:
            try:
                item = next(dataset_iter)
            except StopIteration:
                break
            
            print(item)

            raise ValueError
                
            question = item['question']
            answer = item['answer']['value']

            print(question,answer)
            
            # Process only the first Wikipedia URL
            if item['entity_pages'] and item['entity_pages'][0]['wiki_url']:
                wiki_url = item['entity_pages'][0]['wiki_url']
                
                # Get Wikipedia content
                content = get_wikipedia_content(wiki_url)
                if not content:
                    continue
                
                # Split into paragraphs
                paragraphs = split_into_paragraphs(content)
                if not paragraphs:
                    continue
                
                # Create BM25 index
                bm25, tokenized_paragraphs = create_bm25_index(paragraphs)
                
                # Retrieve top paragraph
                top_results = retrieve_top_paragraph(bm25, tokenized_paragraphs, question, paragraphs)
                
                if top_results:
                    context, score = top_results[0]
                    print(context,"\n\n",score)

                    triplets.append({
                        'question': question,
                        'context': context,
                        'answer': answer,
                        'score': score,
                        'wiki_url': wiki_url
                    })
                    processed_count += 1
                    pbar.update(1)
    
    return triplets

def main():
    # Process dataset and create triplets
    print("Processing TriviaQA dataset...")
    triplets = process_trivia_qa()
    
    # Save results
    print(f"Generated {len(triplets)} question-context-answer triplets")
    
    # Print a sample triplet
    if triplets:
        sample = triplets[0]
        print("\nSample triplet:")
        print(f"Question: {sample['question']}")
        print(f"Context: {sample['context'][:200]}...")
        print(f"Answer: {sample['answer']}")
        print(f"BM25 Score: {sample['score']:.2f}")
        print(f"Wikipedia URL: {sample['wiki_url']}")

if __name__ == "__main__":
    main()