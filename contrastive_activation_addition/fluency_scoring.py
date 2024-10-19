from typing import Dict, List
from openai import AzureOpenAI
import os
import nltk
from langchain.embeddings import SentenceTransformerEmbeddings
from spacy.language import Language
import spacy
from textstat import (
    flesch_reading_ease,
    flesch_kincaid_grade
)

from typing import Union
import numpy as np
from numpy.linalg import norm

nlp = spacy.load("en_core_web_sm")

from evaluate import load

# perplexity = load("perplexity", module_type="metric")

# def get_perplexity(
#     question:str
# ) -> Union[int,float]:
#     """Computes GPT2-Perplexity of question.
#     Args:
#         question (str): question
#     Returns:
#         Union[int, float]: GPT2 perplexity of question
#     """
#     results = perplexity.compute(predictions=[question], model_id='gpt2')
#     return results['perplexities'][0]

def get_reading_ease(
    question: str
) -> Union[int, float]:
    """Computes reading ease of question.
    Args:
        question (str): question
    Returns:
        Union[int, float]: reading ease of question
    """
    return flesch_reading_ease(question)


def get_reading_grade(
    question: str,
) -> Union[int, float]:
    """Computes reading grade of question.
    Args:
        question (str): question
    Returns:
        Union[int, float]: reading grade of question
    """
    return flesch_kincaid_grade(question)


def get_number_of_characters(
    question: str
) -> Union[int, float]:
    """Computes character length of question.
    Args:
        question (str): question
    Returns:
        Union[int, float]: character length of question
    """
    return len(question)

def get_number_of_words(
    question: str,
    nlp: Language
) -> Union[int, float]:
    """Computes word length of question.
    Args:
        question (str): question
        nlp (Language): spacy nlp object
    Returns:
        Union[int, float]: word length of question
    """
    doc = nlp(question)
    return len(doc)

def get_cosine_similarity(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray
) -> float:
    """Computes cosine similarity between two embeddings.
    Args:
        embedding_a (np.ndarray): embedding a
        embedding_b (np.ndarray): embedding b
    Returns:
        float: cosine similarity between two embeddings
    """
    return np.dot(embedding_a, embedding_b) / (norm(embedding_a) * norm(embedding_b))

from typing import Dict, Tuple
import re

def summarize_repetitions_by_length(repetition_info: Dict[Tuple[str, int], int]) -> Dict[int, int]:
    """
    Summarizes the repetition information by phrase length.

    Args:
        repetition_info (Dict[Tuple[str, int], int]): The output from get_repetition_info

    Returns:
        Dict[int, int]: A dictionary where keys are phrase lengths and values are total number of repeats
    """
    summary = {}
    for (_, phrase_length), repeat_count in repetition_info.items():
        summary[phrase_length] = summary.get(phrase_length, 0) + repeat_count - 1
    
    return summary

def get_repetition_info(question: str) -> Dict[Tuple[str, int], int]:
    """Computes the repetition information for the given question.

    Args:
        question (str): question
    Returns:
        Dict[Tuple[str, int], int]: A dictionary where keys are tuples of (repeated_phrase, length_of_phrase)
                                    and values are the number of repetitions
    """
    # Normalize the text
    question = question.strip("<|eot_id|>")
    text = question.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    repetition_info = {}
    max_phrase_length = len(words) // 2  # Maximum phrase length to check

    for phrase_length in range(1, max_phrase_length + 1):
        i = 0
        while i < len(words) - phrase_length:
            phrase = tuple(words[i:i+phrase_length])
            repeat_count = 1
            end = i + phrase_length

            # Check for successive repetitions
            while end + phrase_length <= len(words) and tuple(words[end:end+phrase_length]) == phrase:
                repeat_count += 1
                end += phrase_length

            if repeat_count > 1:
                repetition_info[(' '.join(phrase), phrase_length)] = repeat_count
                words[i:end] = list(phrase)
                i = 0
            else:
                i += 1

    return summarize_repetitions_by_length(repetition_info)

# Test cases
# test_cases = [
#     "He had had had to had to had to work had to work had to work",
#     "He was going to the library when he saw a guy that was going to the library, when he saw a guy that was going to the library, when he saw a guy that was going to the library,...",
#     "This is a normal sentence without any repetition.",
#     "The cat sat on the mat. The cat sat on the mat. The dog chased the cat.",
#     "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo",
# ]

# for i, case in enumerate(test_cases, 1):
#     print(f"Test case {i}:")
#     result = get_repetition_info(case)
#     for length, count in result.items():
#         print(f"\tLength: {length}, Repetitions: {count}")
#     print()


# client = AzureOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
#     api_version="2024-02-01",
#     azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
# )

# FLUENCY_PROMPTS = {
#     "fluency": """Please provide the answer with a score of 0 or 1 assessing the fluency of the text.
#     - Assess how smoothly the text reads.
#     - A 1 means ideas are presented in an easy-to-read, understandable fashion.
#     - A 0 should be given if any one sentence is hard to interpret due to its language.""",
#     "coherence":"""Please provide the answer with a score of 0 or 1 assessing the coherence of the text.
#     - Evaluate the logical connection between ideas.
#     - Assess the overall structure and organization of the text.
#     - A 1 means ideas are ordered in a logical way and connected properly.
#     - A 0 implies that the connection between any two ideas is difficult to interpret.""",
#     "grammar":"""Please provide the answer with a score of 0 or 1 assessing the grammar of the text.
#     - Identify any grammatical errors and awkward constructions.
#     - A 1 represents perfect content with no grammatical, spelling or punctuation errors.
#     - A 0 implies at least one grammatical error or spelling mistake."""

# }

# def make_prompts(type,question,answer):
#     system_prompt = "You are an evaluator for a language models' answers to questions. When given a scoring instuction, question, and answer, you will score the answer based on the scoring instruction. You only ever return a numerical score and no other text."
#     user_prompt = f"{FLUENCY_PROMPTS[type]}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
#     return system_prompt,user_prompt

# def make_gpt4_request(system_prompt, user_prompt) -> str:
#     deployment_name='gpt-4o'
#     response = client.chat.completions.create(
#         model=deployment_name, 
#         messages=[
#                 {
#                     "role": "system",
#                     "content": system_prompt,
#                 },
#                 {"role": "user", "content": user_prompt},
#             ],
#         max_tokens=10,
#         temperature=0.0,
#     )
    
#     return response.choices[0].message.content

# def get_score(d,type): # type = "fluency","coherence","grammar"
#     ques = d['question']
#     ans = d['model_output']
#     system_prompt, user_prompt = make_prompts(type,ques,ans)
#     score = make_gpt4_request(system_prompt, user_prompt)
#     try:
#         numeric_score = float(score)
#     except Exception:
#         print(f"Error scoring. Prompt: {user_prompt}, Response: {score}")
    
#     return numeric_score

def get_score(d,type): # type = "reading_ease","reading_grade","num_chars","num_words","perplexity"
    ques = d['question']
    
    ans = d['model_output']
    if ans.endswith("<|eot_id|>"):
        ans = ans[:-len("<|eot_id|>")]

    if type=="reading_ease":
        reading_ease = get_reading_ease(ans)
        return reading_ease
    elif type=="reading_grade":
        reading_grade = get_reading_grade(ans)
        return reading_grade
    elif type=="num_chars":
        num_chars = get_number_of_characters(ans)
        return num_chars
    elif type=="num_words":
        num_words = get_number_of_words(ans,nlp)
        return num_words
    elif type=="perplexity":
        perplexity = get_perplexity(ans)
        return perplexity