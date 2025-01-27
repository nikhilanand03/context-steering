import jsonlines
from tqdm import tqdm
import json

INPUT_FILE = "v1.0-simplified_nq-dev-all.jsonl"

final_list = []

with jsonlines.open(INPUT_FILE) as f:
    for i,obj in tqdm(enumerate(f)):
        qas_id = obj['example_id']
        ques = obj['question_text']

        li_of_contexts = []

        for annot in obj['annotations']:
            if len(annot['short_answers'])==0:
                continue

            long_ans_ST = annot['long_answer']['start_token']
            long_ans_ET = annot['long_answer']['end_token']

            long_answer_span = ' '.join([
                item["token"]
                for item in obj["document_tokens"][long_ans_ST:long_ans_ET]
            ])

            short_answers = []

            for short_ans in annot['short_answers']:
                short_ans_ST = short_ans['start_token']
                short_ans_ET = short_ans['end_token']

                short_answer_span = ' '.join([
                    item["token"]
                    for item in obj["document_tokens"][short_ans_ST:short_ans_ET]
                ])

                short_answers.append(short_answer_span)

            li_of_contexts.append({
                'gold_ctx': long_answer_span,
                'short_answers': short_answers,
                'avg_length_short_ans': sum(len(ans) for ans in short_answers) / len(short_answers) if short_answers else 0
            })

        # Converting li_of_contexts into a single gold_ctx, short_answers pair. short_answers is still a list but from one annotator.
        gold_ctx_FINAL = None

        for i,context in enumerate(li_of_contexts):
            if context['gold_ctx'][:3]=="<P>":
                gold_ctx_FINAL=context['gold_ctx']
                break
        
        if gold_ctx_FINAL is None:
            try:
                min_ctx_len = len(li_of_contexts[0]['gold_ctx'])

                for context in li_of_contexts:
                    if len(context['gold_ctx']) <= min_ctx_len:
                        gold_ctx_FINAL = context['gold_ctx']
            except:
                pass
        
        short_answers_FINAL = None

        try:
            min_avg_length = li_of_contexts[0]['avg_length_short_ans']

            for i, context in enumerate(li_of_contexts):            
                if context['avg_length_short_ans'] <= min_avg_length:
                    short_answers_FINAL = context['short_answers']
        except:
            pass
        
        final_item = {
            'question': ques,
            'gold_ctx': gold_ctx_FINAL,
            'short_answers': short_answers_FINAL
        }

        final_list.append(final_item)

with open("nq_dev_gold_ctx.json",'w') as f:
    json.dump(final_list,f)

