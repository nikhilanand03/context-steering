import json
from tqdm import tqdm
import os

def convert_triviaqa_to_squad_inp_format(input_data, domain='Web', version='1.0', verified_eval=False):
    contexts_dir = os.path.join(os.getcwd(), 'contexts')
    os.makedirs(contexts_dir, exist_ok=True)
    
    converted_data = {
        'Domain': domain,
        'Version': version,
        'VerifiedEval': verified_eval,
        'Split': 'validation',
        'Data': []
    }


    for entry in tqdm(input_data):
        # Prepare entity pages
        entity_pages = []
        if entry.get('entity_pages') and entry['entity_pages'].get('filename'):
            for filename, context in zip(
                entry['entity_pages']['filename'], 
                entry['entity_pages'].get('wiki_context', [''] * len(entry['entity_pages']['filename']))
            ):
                # Save context to file
                context_filepath = os.path.join(contexts_dir, filename)
                os.makedirs(os.path.dirname(context_filepath), exist_ok=True)
                with open(context_filepath, 'w', encoding='utf-8') as f:
                    f.write(context)

                entity_pages.append({
                    'Filename': filename, 
                    'DocPartOfVerifiedEval': verified_eval
                })

        # Prepare search results
        search_results = []
        if entry.get('search_results') and entry['search_results'].get('filename'):
            for filename, context in zip(
                entry['search_results']['filename'], 
                entry['search_results'].get('search_context', [''] * len(entry['search_results']['filename']))
            ):
                # Save context to file
                context_filepath = os.path.join(contexts_dir, filename)
                os.makedirs(os.path.dirname(context_filepath), exist_ok=True)
                with open(context_filepath, 'w', encoding='utf-8') as f:
                    f.write(context)

                search_results.append({
                    'Filename': filename, 
                    'DocPartOfVerifiedEval': verified_eval
                })


        answer = {
            'NormalizedValue': entry['answer']['normalized_value'],
            'NormalizedAliases': entry['answer']['normalized_aliases']
        }

        question_entry = {
            'QuestionId': entry['question_id'],
            'Question': entry['question'],
            'Answer': answer,
            'EntityPages': entity_pages,
            'SearchResults': search_results,
            'QuestionPartOfVerifiedEval': verified_eval
        }

        converted_data['Data'].append(question_entry)

    return converted_data

def save_triviaqa_json(converted_data, output_file):
    """
    Save the converted TriviaQA data to a JSON file.
    
    Parameters:
    - converted_data: Converted TriviaQA dataset dictionary
    - output_file: Path to save the output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

# Example usage
def main():
    with open("kilt_triviaqa_subset.json",'r') as f:
        input_data = json.load(f)

    converted_data = convert_triviaqa_to_squad_inp_format(
        input_data, 
        domain='Web', 
        version='1.0', 
        verified_eval=True
    )

    save_triviaqa_json(converted_data, 'converted_kilt_triviaqa_subset.json')

if __name__ == '__main__':
    main()