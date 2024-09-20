import json
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from tqdm import tqdm
import re

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # or your deployment
    api_version="2024-02-01",  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

SYSTEM_PROMPT = """You are an AI Assistant tasked with refining paragraphs from the NQ-SWAP dataset. This dataset contains Wikipedia paragraphs where a specific named entity has been replaced with a random entity. Your task is to:

1. Identify the random entity (it will be the one that deviates from factual accuracy).
2. Replace any subsequent mentions of the original named entity with this random entity.
3. If there are temporal inconsistencies due to the entity swap, adjust years to be consistent with the random entity's era. Use similar time frames, not exact years.
4. Make no other changes to the paragraph.

Output format:
- If changes are needed, use: <P>...corrected paragraph...</P>
- If the paragraph is already consistent, respond only with: Yes

Important:
- NEVER alter the random entity.
- Only make the specific changes outlined above.
- Preserve the original wording and structure as much as possible.

Please process the provided paragraph accordingly."""

SYSTEM_PROMPT_YEAR = """Task: Analyze the given text for mentions of multiple years.

Instructions:
1. Read the provided paragraph carefully.
2. Identify all years mentioned in the text.
3. Compare each pair of years.
4. If any two years differ by more than 500 years, respond with 'Yes'.
5. If no such pair is found, or if fewer than two years are mentioned, respond with 'No'.

Example:
Input: "The Roman Empire emerged in 476 AD, while it collapsed in 1900."
Output: Yes

Input: "World War I began in 1914 and ended in 1918."
Output: No

Please provide your answer based on these instructions. PLEASE PROVIDE ONLY SINGLE-WORD ANSWERS.
"""

def ask_gpt(system,prompt):
    messages = [
        ("system", system),
        ("human", prompt)
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content

def load_data():
    with open("generate_dataset_long.json",'r') as file:
        data = json.load(file)

    return data

def fix_name_problem(i,data):
    pass

def clean_name_inconsistencies(data):
    name_problem_ids = []
    tag_problem_ids = []
    data_new = []
    j = 0
    for i in tqdm(range(len(data))):
        d = data[i]
        qi,ami,anmi = d['question'],d['answer_matching_behavior'],d['answer_not_matching_behavior']
        question,options = tuple(d['question'].split("\n\n"))

        split_word = "</"+question[question.index("<")+1:question.index(">")]+">"
        try:
            context,question = tuple(item for item in question.split(split_word))
        except:
            tag_problem_ids.append((i,j*5+2))
            continue
        context = context + split_word
        question = question.strip()
        
        optionA = options[options.index("(A)")+4:options.index("(B)")-2]
        optionB = options[options.index("(B)")+4:]
        
        contextOpt = optionA if d['answer_matching_behavior']=="(A)" else optionB
        memoryOpt = optionB if d['answer_matching_behavior']=="(A)" else optionA

        memoryWords = memoryOpt.split()

        if len(memoryWords)==2:
            if memoryWords[1] in context and len(memoryWords[1])>1:
                name_problem_ids.append((i,j*5+2,memoryWords[1]))
                try:
                    qi = context.replace(memoryWords[1],contextOpt.split()[1])+" "+question+"\n\n"+options
                except:
                    print("[ALERT] Not replaced for: ",i,"**",j*5+2,"**",contextOpt,"**",memoryOpt)
                    continue

        data_new.append({"question":qi,"answer_matching_behavior":ami,"answer_not_matching_behavior":anmi})
        j+=1
        
    print("NAME_PROBLEM_IDS (Cleaned): ",end="")
    print(len(name_problem_ids))
    # print(name_problem_ids)
    print("TAG_PROB_IDS:",end="")
    print(len(tag_problem_ids))

    return data_new

def clean_year_inconsistencies(data):
    year_prob_ids = []
    data_new = []
    for i in tqdm(range(len(data))):
        d = data[i]
        question,_ = tuple(d['question'].split("\n\n"))
        split_word = "</"+question[question.index("<")+1:question.index(">")]+">"
        context,_ = tuple(item for item in question.split(split_word))
        context = context + split_word
        
        if has_two_four_digit_numbers(context):
            # print(context)
            ans = ask_gpt(SYSTEM_PROMPT_YEAR,context)
            # print(ans)
            if ans=="Yes":
                year_prob_ids.append(i)
    
    print("YEAR_PROB_IDS:",len(year_prob_ids))
    print(year_prob_ids)

    for i in range(len(data)):
        if i not in year_prob_ids:
            data_new.append(data[i])

    return data_new


def clean_table_tags(data):
    data_new = []
    for i in range(len(data)):
        if not data[i]['question'].startswith("<Table>"):
            data_new.append(data[i])

    print("Cleaned table tags: ",len(data_new))
    return data_new

def save_data_current(data):
    print("Saving data, size = ",len(data))
    with open("generate_dataset_cleaned_contexts.json",'w') as file:
        json.dump(data,file)
    
def has_two_four_digit_numbers(text):
    words = text.split()
    count = 0
    
    for word in words:
        word = word.strip(".,!?;:")
        
        if word.isdigit() and len(word) == 4:
            count += 1
    
        if count > 1:
            return True
    
    return False

if __name__=="__main__":
    data = load_data()
    data = clean_table_tags(data)
    data = clean_name_inconsistencies(data)
    data = clean_year_inconsistencies(data)
    save_data_current(data)
