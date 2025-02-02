import json
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
import nltk
from tqdm import tqdm
import csv
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

nltk.download('punkt')

def get_embeddings(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings

def get_wikipedia_paragraphs(title):
    """Fetch Wikipedia HTML page and return list of paragraphs."""
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch: {title}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
    
    return paragraphs

def get_top_1_paragraph(question, all_paragraphs, retriever='bm25'): # retriever in ['bm25','contriever']
    if retriever=='bm25':
        tokenized_paragraphs = [nltk.word_tokenize(p.lower()) for p in all_paragraphs]
        tokenized_question = nltk.word_tokenize(question.lower())

        bm25 = BM25Okapi(tokenized_paragraphs)

        scores = bm25.get_scores(tokenized_question)
        best_idx = scores.argmax()

        return all_paragraphs[best_idx]
    elif retriever=='contriever':
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        model = AutoModel.from_pretrained('facebook/contriever')
        
        question_embedding = get_embeddings([question])
        paragraph_embeddings = get_embeddings(all_paragraphs)
        
        similarities = torch.matmul(question_embedding, paragraph_embeddings.T)
        
        best_idx = similarities.squeeze().argmax().item()
        
        return all_paragraphs[best_idx]
    
    return None

def save_retrieved_contexts(input_file, output_file, retriever='bm25'):
    with open(input_file, "r", encoding="utf-8") as f:
        li_of_items = []

        for i,line in tqdm(enumerate(f)):
            all_paragraphs = []
            data = json.loads(line)
            s_title = data.get("s_wiki_title")
            # o_title = data.get("o_wiki_title")
            
            if s_title:
                all_paragraphs.extend(get_wikipedia_paragraphs(s_title))
            # if o_title:
            #     all_paragraphs.extend(get_wikipedia_paragraphs(o_title))
            
            # print(all_paragraphs)
            if retriever == 'bm25':
                context = get_top_1_paragraph(data['question'], all_paragraphs, 'bm25')
            elif retriever == 'contriever':
                context = get_top_1_paragraph(data['question'], all_paragraphs, 'contriever')
            
            li_of_items.append({
                'question': data['question'],
                'ctxs': [context]
            })
        
    with open(output_file, "w", encoding="utf-8") as f:
        for item in li_of_items:
            f.write(json.dumps(item) + "\n")

def save_tsv_input_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        li_of_items = []
        for line in f:
            data = json.loads(line)
            li_of_items.append({
                "question": data['question'],
                "answers": eval(data["possible_answers"]),
                "ans": eval(data["possible_answers"])[0]
            })
    
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=li_of_items[0].keys(), delimiter="\t")
        writer.writeheader()  # Write column headers
        writer.writerows(li_of_items)

def main():
    save_tsv_input_file("popqa_raw.jsonl", "popqa_test.tsv")
    # save_retrieved_contexts("popqa_raw.jsonl","popqa_bm25_results.jsonl",'bm25')
    save_retrieved_contexts("popqa_raw.jsonl","popqa_contriever_results.jsonl",'contriever')


if __name__=="__main__":
    main()