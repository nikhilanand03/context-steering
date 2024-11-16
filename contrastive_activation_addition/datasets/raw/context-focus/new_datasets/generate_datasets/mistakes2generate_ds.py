import json
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
import random
from tqdm import tqdm

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

# START,END = 0,2500*8
START,END = 2500*8,3125*8

with open("hotpot_training_data.json",'r') as file:
    data = json.load(file)[START:END]

SYSTEM_PROMPT = """
You are an AI assistant tasked with generating a concise, grammatically-correct sentence answer from a given question and a short answer to that question.

Your goal is to combine the provided question and answer into a single, coherent sentence that clearly conveys the information.

You will be given a question and a corresponding short answer. Your task is to return a single string containing the generated sentence answer.

Here are some examples:

Question: Who is the President of the US?
Answer: Donald Trump
Output: Donald Trump is the President of the US.

Question: What was the nationality of the drummer of "I'm missing you"?
Answer: American
Output: The nationality of the drummer of "I'm missing you" was American.

Question: When did the Berlin Wall fall?
Answer: 1989
Output: The Berlin Wall fell in 1989.

Your response should be a single string containing the generated sentence answer.
"""

def generate_sentence_from_QA(question,answer):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Please process the following:\n\nQuestion: {question}; Answer: {answer}")
    ]

    response = llm.invoke(messages)
    
    return response.content


def convert_context_to_rag(context):
    rag_contexts = []
    for rag_item in context:
        title = rag_item[0]
        sentences = rag_item[1]
        rag_context = "".join(sentences)
        rag_contexts.append(rag_context)
    return rag_contexts

# scores_filename = "scores_hotpot_results_correct_3k.json"
# scores_filename = "scores_0_2500_hotpot_results_100k.json"
scores_filename = "scores_2500_3125_hotpot_results_100k.json"
with open(scores_filename,'r') as file:
    data_scores_temp = json.load(file)

data_scores = []

for row in data_scores_temp:
    for item in row:
        data_scores.append(item)

mistakes_dataset = []

for i in tqdm(range(len(data_scores))):
    if data_scores[i]['score']=='1':
        continue

    item = data[i]
    item_scores = data_scores[i]

    option_correct = generate_sentence_from_QA(item['question'],item['answer'])
    option_wrong = item_scores['model_output'][:-10]

    answer_matching_behavior = "(A)" if random.choice([0, 1]) == 0 else "(B)"
    answer_not_matching_behavior = "(B)" if answer_matching_behavior=="(A)" else "(A)"

    if answer_matching_behavior=="(A)":
        options = [option_correct, option_wrong]
    else:
        options = [option_wrong, option_correct]

    item_final = {
        "question": item_scores['question'],
        "answer": item_scores['answer'],
        "rag": convert_context_to_rag(item['context']),
        "options": options,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior
    }

    mistakes_dataset.append(item_final)

with open("final_generate_dataset.json",'w') as f:
    json.dump(mistakes_dataset,f)