import json

CONTEXT_FAILURE_INDICES = [4694, 4689, 4686, 4685, 4675, 4643, 4633, 4632, 4631, 4630, 4627, 4623, 4621, 4616, 4608, 4606, 4601, 4575, 4572, 4567, 4566, 4564, 4563, 4561, 4558, 4556, 4555, 4554, 4549, 4548, 4536, 4535, 4533, 4532, 4531, 4524, 4522, 4521, 4520, 4519, 4517, 4514, 4512, 4507, 4497, 4490, 4486, 4485, 4480, 4477, 4476, 4475]

def get_failure_test_from_long(long_path):
    with open(long_path,'r') as f:
        data = json.load(f)
    
    test_data = []
    
    for id in CONTEXT_FAILURE_INDICES:
        test_data.append(data[id])
    
    return test_data

def convert_cq_format(in_path,out_path):
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
        prompt_new = f"Context: {context}\nQuestion: {question}"

        out_list.append({"question":prompt_new, "answer_matching_behavior":d['answer_matching_behavior'], "answer_not_matching_behavior":d['answer_not_matching_behavior']})

    with open(out_path,'w') as f:
        json.dump(out_list,f)