import json

def convert_cq_format(in_path,out_path,remove_ab = False):
    with open(in_path,'r') as f:
        data = json.load(f)
    
    out_list = []

    for d in data:
        prompt = d["question"]
        tag_start = prompt.index("<")
        tag_end = prompt.index(">")+1
        tag = prompt[tag_start:tag_end]
        antitag = "</"+tag[1:-1]+">"
        context,question = prompt.split(antitag)[0]+antitag,prompt.split(antitag)[1].strip()
        if remove_ab: question = question.split("\n\nChoices:")[0]
        prompt_new = f"Context: {context}\nQuestion: {question}"

        if not remove_ab: out_list.append({"question":prompt_new, "answer_matching_behavior":d['answer_matching_behavior'], "answer_not_matching_behavior":d['answer_not_matching_behavior']})
        else: 
            if "sub_answer" in d:
                out_list.append({"question":prompt_new, "sub_answer":d['sub_answer']})
            else:
                out_list.append({"question":prompt_new})

    with open(out_path,'w') as f:
        json.dump(out_list,f)

def remove_options(in_path,out_path):
    with open(in_path,'r') as f:
        data = json.load(f)

    new_li = []

    for item in data:
        del item['options']
        del item['answer_matching_behavior']
        del item['answer_not_matching_behavior']

        new_li.append(item)

    with open(out_path,'w') as f:
        json.dump(new_li,f)