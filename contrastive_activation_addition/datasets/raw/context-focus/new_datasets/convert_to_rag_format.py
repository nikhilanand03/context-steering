import json

final_list = []

with open("multihopqa.json",'r') as f:
    li = json.load(f)
    rows = li[0]['rows']

    for row in rows:
        context = "[Document 1]: "
        contexts = row['row']['context']['content']

        for i,item in enumerate(contexts):
            for sentence in item:
                if not context == "":
                    context = context + " "
                context = context + sentence
            
            if not i==len(contexts)-1:
                context = context + f"\n\n[Document {i+2}]: "

        question = row['row']["question"]
        system = "You are a helpful assistant. Based on the following documents, provide a concise answer. Your answer should be short, containing only a single word or phrase representing the answer. Avoid providing sentence answers."
        combined = f"{system}\n\n{context}\n\nQuestion: {question}"

        final_list.append({"question": combined, "answer": row['row']['answer']})

with open("rag_format_multihopqa.json",'w') as f:
    json.dump(final_list,f)
