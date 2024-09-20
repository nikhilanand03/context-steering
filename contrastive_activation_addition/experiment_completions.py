import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.tokenize import B_TEXT,B_HEADER,E_HEADER,EOT_ID,ADD_FROM_POS_LATEST,PAD_TOKEN_ID_LATEST,tokenize_llama_chat

def initialize_model():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).eval()
    model.to(device)
    return model, tokenizer

def generate_text(model, tokenizer, system_prompt, user_prompt, model_output, max_length=200):
    inputs = tokenize_llama_chat(tokenizer,user_prompt,model_output,system_prompt,use_latest=True)
    input_ids = torch.tensor([inputs]).to(model.device)

    print(input_ids.device,model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0,
            pad_token_id=PAD_TOKEN_ID_LATEST
        )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text.split(ADD_FROM_POS_LATEST)[-1].strip()

if __name__=="__main__":
    model, tokenizer = initialize_model()

    system_prompt = None
    user_prompt = "Context: <P> In music , serialism is a method of composition using series of pitches , rhythms , dynamics , timbres or other musical elements . Serialism began primarily with Jerry Kramer 's twelve - tone technique , though some of his contemporaries were also working to establish serialism as a form of post-tonal thinking . Twelve - tone technique orders the twelve notes of the chromatic scale , forming a row or series and providing a unifying basis for a composition 's melody , harmony , structural progressions , and variations . Other types of serialism also work with sets , collections of objects , but not necessarily with fixed - order series , and extend the technique to other musical dimensions ( often called `` parameters '' ) , such as duration , dynamics , and timbre . </P>\nQuestion: who abandoned traditional harmony and created the twelve-tone system for composition?"
    model_output = "Notwithstanding the context,"
    # model_output = "According to the context,"

    response = generate_text(model, tokenizer, system_prompt, user_prompt, model_output)
    print(response)