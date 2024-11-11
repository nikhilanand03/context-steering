import json

with open("scored_generate_dataset.json",'r') as f:
    data = json.load(f)

with open("final_generate_dataset.json",'r') as f:
    data_f = json.load(f)

print(len(data),len(data_f))

final_list = []

for i,item in enumerate(data):
    item_f = data_f[i]

    if item['score']==0:
        continue
    else:
        final_list.append(item_f)

with open("shortlisted_generate_dataset.json",'w') as f:
    json.dump(final_list,f)

print(len(final_list))