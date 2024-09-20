import json

def run_pipeline():
    with open("all_ab_cleaned_contexts.json",'r') as file:
        all_ab_data = json.load(file)

    with open("test_dataset_open_ended_successes.json",'r') as file:
        test_oe_data = json.load(file)
    
    test_ab_data = []
    
    all_ab_data = all_ab_data[::-1]

    j=0

    for i in range(len(all_ab_data)):
        if j>=len(test_oe_data):
            break
        d = all_ab_data[i]
        ques = d['question']
        tag = ques[:ques.index(">")+1]
        end_tag = "</"+tag[1:-1]+">"
        
        context_start = ques.index(tag)
        context_end = ques.index(end_tag)+len(end_tag)
        context = ques[context_start:context_end].strip()

        if context in test_oe_data[j]['question']:
            j+=1
            test_ab_data.append(d)
        
    with open("test_dataset_ab_successes.json",'w') as file:
        json.dump(test_ab_data,file)

if __name__=="__main__":
    run_pipeline()