import json
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
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

SYSTEM_PROMPT_1 = """You are an AI assistant tasked with a simple evaluation. You will receive a sentence or paragraph.

Your only task is to determine if the text indicates that 'the provided contexts/context are/is insufficient to answer the question' or similar meaning. This includes variations like:
- 'I cannot find the answer in the given context'
- 'The context does not provide this information'
- 'Based on the available context, I cannot determine...'
- 'The context is insufficient to answer...'
- 'There is no information in the context about...'
- 'Without additional context, I cannot...'

Output 1 if the text suggests context insufficiency in any way, and 0 otherwise.

Your response should be exactly one number (1 or 0) with no additional explanation or text."""

SYSTEM_PROMPT_2 = """You are an AI assistant tasked with evaluating the quality of model-generated answers. You will receive:
1. A question
2. A ground truth answer
3. Supporting contexts
4. A model's output answer

Your task is to determine if the model's output either:
a) Contradicts the provided context
b) Fails to answer the question

Score 1 if either condition is met, and 0 if the answer is acceptable.

Focus only on these specific criteria - do not evaluate grammar, completeness, or other aspects of the response.

Here are some examples:

Example 1:
{
    'question': 'When was the Eiffel Tower built?',
    'ground_truth_answer': 'The Eiffel Tower was completed in 1889.',
    'contexts': ['The Eiffel Tower was constructed from 1887 to 1889.', 'The tower was completed for the 1889 World's Fair in Paris.'],
    'model_output': 'The Eiffel Tower was built in 1878.'
}
Score: 1
Reason: The model's answer contradicts the context, which clearly states construction was from 1887-1889.

Example 2:
{
    'question': 'What is the capital of France?',
    'ground_truth_answer': 'Paris is the capital of France.',
    'contexts': ['Paris has been France's capital since 508 CE.', 'As the capital city, Paris is France's political and economic center.'],
    'model_output': 'France is a beautiful country with many historic landmarks.'
}
Score: 1
Reason: The model's output fails to answer the specific question about France's capital.

Example 3:
{
    'question': 'How tall is Mount Everest?',
    'ground_truth_answer': 'Mount Everest is 29,029 feet (8,848 meters) tall.',
    'contexts': ['Mount Everest stands at 29,029 feet (8,848 meters) above sea level.', 'It is Earth's highest mountain above sea level.'],
    'model_output': 'Mount Everest reaches a height of 29,029 feet.'
}
Score: 0
Reason: The answer is correct and aligned with the context, directly answering the question.

For each evaluation, you should:
1. Carefully read the question and context
2. Check if the model's output contradicts any information in the context
3. Verify if the output actually answers the question asked
4. Provide a score (1 or 0)
5. Briefly explain your reasoning

Your response should follow this format:
Score: [0 or 1]
Reason: [Brief explanation]
"""

def generate_GPT_output(system_prompt,item):
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please process the following:\n\n{item}")
    ]

    response = llm.invoke(messages)
    
    return response.content

with open("supporting_facts_generate_dataset.json",'r') as f:
    data = json.load(f)

final_list = []

for item in tqdm(data):
    item_to_add = item.copy()

    resp1 = generate_GPT_output(SYSTEM_PROMPT_1,item['model_output'])
    if resp1 == '1':
        item_to_add['score'] = 2
        final_list.append(item_to_add)
        continue

    item_copy = item.copy()
    item_copy['ground_truth_answer'] = item_copy.pop('answer')
    item_copy['contexts'] = item_copy.pop('supporting_contexts')

    resp2 = generate_GPT_output(SYSTEM_PROMPT_2,item_copy)
    if resp2.split("\n")[0].split(" ")[1] == '1':
        item_to_add['score'] = 1
        final_list.append(item_to_add)
        continue

    item_to_add['score'] = 0
    final_list.append(item_to_add)

with open("scored_generate_dataset.json",'w') as f:
    json.dump(final_list,f)