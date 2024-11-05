import json

final_list = []

with open("multihopqa.json",'r') as f:
    li = json.load(f)
    rows = li[0]['rows']

    for row in rows:
        context = ""
        contexts = row['row']['context']['content']

        for item in contexts:
            for sentence in item:
                if not context == "":
                    context = context + " "
                context = context + sentence
        
        context = f"<P> {context} </P>"
        question = row['row']["question"]

        combined = f"Context: {context}\nQuestion: {question}"

        final_list.append({"question": question, "context": context, "answer": row['row']['answer']})

with open("final_multihopqa.json",'w') as f:
    json.dump(final_list,f)

### MUSIQUE

final_list = []

with open("musique.json",'r') as f:
    li = json.load(f)
    rows = li[0]['rows']

    for row in rows:
        context_all = f"<P> {row['row']['text_all']} </P>"
        context_all_support = f"<P> {row['row']['text_all_support']} </P>"
        question = row['row']['question']
        answer = row['row']['answer']

        final_list.append({"question": question, "context_all": context_all, "context_support": context_all_support, "answer": answer})

with open("final_musique.json",'w') as f:
    json.dump(final_list,f)

### NQ

final_list = []

with open("nq.json",'r') as f:
    li = json.load(f)
    rows = li[0]['rows']

    for row in rows:
        context = f"<P> {','.join(row['row']['retrieved_passages'])} </P>"
        final_list.append({"question": row['row']['question'], "long_ans": row['row']['long_answers'], "short_ans": row['row']['short_answers'], "context": context})

with open("final_nq.json",'w') as f:
    json.dump(final_list,f)

### HotpotQA

final_list = []

with open("hotpotqa.json",'r') as f:
    li = json.load(f)
    rows = li[0]['rows']

    for i,row in enumerate(rows):
        # print(i)
        context = f"<P> {' '.join(row['row']['rag'])} </P>"
        final_list.append({"question": row['row']['question'], "answer": row['row']['answer'], "context": context})

with open("final_hotpotqa.json",'w') as f:
    json.dump(final_list,f)

### TriviaQA

final_list = []

with open("triviaqa.json",'r') as f:
    li = json.load(f)
    rows = li[0]['rows']

    for i,row in enumerate(rows):
        context = f"<P> {' '.join(row['row']['retrieved_passages'])} </P>"
        final_list.append({"question": row['row']['question'], "answer": row['row']['answer'], "context": context})

with open("final_triviaqa.json",'w') as f:
    json.dump(final_list,f)

### TruthfulQA

final_list = []

with open("truthfulqa.json",'r') as f:
    li = json.load(f)
    rows = li[0]['rows']

    for i,row in enumerate(rows):
        context = f"<P> {' '.join(row['row']['retrieved_passages'])} </P>"
        final_list.append({"question": row['row']['question'], "choices": row['row']['mc1_targets']['choices'], "choice_labels": row['row']['mc1_targets']['labels'], "context": context, "source": row['row']['source']})

with open("final_truthfulqa.json",'w') as f:
    json.dump(final_list,f)
