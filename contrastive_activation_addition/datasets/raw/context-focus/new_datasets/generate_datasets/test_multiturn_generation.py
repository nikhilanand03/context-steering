from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_response(model, tokenizer, conversation, max_length=512):
    # Encode the conversation
    inputs = tokenizer(conversation, return_tensors="pt", add_special_tokens=True)
    
    # Generate response
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=0,
            num_return_sequences=1,
            stopping_criteria=None  # Will stop at EOS token
        )
    
    # Decode the response
    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    return response

def main():
    # Initialize model and tokenizer
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Define the multi-turn conversation
    conversation = """<|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

How can I learn to play chess?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

To start learning chess, I recommend these steps:

1. Learn how each piece moves
2. Understand basic chess rules
3. Practice basic checkmate patterns
4. Study opening principles
5. Play regularly against others

Would you like me to explain any of these in more detail?<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Yes, please explain the basic checkmate patterns.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Here are some fundamental checkmate patterns:

1. Back-rank mate
2. Scholar's mate
3. Smothered mate
4. Queen and king mate

Let me explain the back-rank mate in detail...<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Please continue explaining the back-rank mate.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

"""

    # Generate and print response
    response = generate_response(model, tokenizer, conversation)
    print("\nModel Response:")
    print(response)

if __name__ == "__main__":
    main()