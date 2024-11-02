import json
from typing import List, Dict, Any, Iterator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import logging
from pathlib import Path
import gc
import ijson  # For streaming JSON processing

# Constants
B_HEADER, E_HEADER = "<|start_header_id|>", "<|end_header_id|>"
EOT_ID = "<|eot_id|>"
BATCH_SIZE = 8  # Adjust based on your GPU memory
SAVE_FREQUENCY = 100  # Save results every N examples

class HotpotQAProcessor:
    def __init__(self, model_name: str = "meta-llama/LLaMA-3.1-8b"):
        """Initialize the HotPot QA processor with model and tokenizer."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16  # Use half precision to save memory
            )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def template_prompt(self, documents: List[str], question: str) -> str:
        """Create a formatted prompt from documents and question."""
        system_prompt = "You are a Contextual QA Assistant. Use the following retrieved contexts to answer any questions that may follow."
        
        # Build prompt components
        header = f"{B_HEADER}system{E_HEADER}\n\n{system_prompt}{EOT_ID}"
        
        context_input = "\n\n".join(
            f"[Document {i+1}]: {doc}" 
            for i, doc in enumerate(documents)
        )
        
        user_context = f"{B_HEADER}user{E_HEADER}\n\n{context_input.strip()}{EOT_ID}"
        assistant_reply = f"{B_HEADER}assistant{E_HEADER}\n\nWill do! I'll use these contexts to answer your questions.{EOT_ID}"
        final_question = f"{B_HEADER}user{E_HEADER}\n\n{question.strip()}{EOT_ID}\n{B_HEADER}assistant{E_HEADER}\n\n"
        
        return f"{header}{user_context}{assistant_reply}{final_question}"

    def stream_json_data(self, input_path: str) -> Iterator[Dict]:
        """Stream JSON data to avoid loading entire file into memory."""
        with open(input_path, 'rb') as file:
            parser = ijson.items(file, 'item')
            for item in parser:
                yield item

    def save_batch_results(self, results: List[Dict], output_path: Path, mode: str = 'a'):
        """Save batch results to file, either appending or writing new file."""
        try:
            if mode == 'w' or not output_path.exists():
                # If first write, include opening bracket
                with open(output_path, mode) as f:
                    f.write('[\n')
            else:
                # If appending, first remove the closing bracket from file
                with open(output_path, 'rb+') as f:
                    f.seek(-2, 2)  # Go to second-to-last byte
                    f.truncate()
                    f.close()
                
                # Append results with comma separator
                with open(output_path, 'a') as f:
                    f.write(',\n')
            
            # Write results
            with open(output_path, 'a') as f:
                json.dump(results, f, indent=2)
                if mode == 'a':
                    f.write('\n]')
                    
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise

    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of HotPot QA examples."""
        prompts = []
        for item in batch_data:
            docs = ["".join(context[1]) for context in item['context']]
            prompt = self.template_prompt(docs, item['question'])
            prompts.append(prompt)

        # Tokenize all prompts in batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Generate responses for batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )

        # Decode outputs and free memory
        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        del inputs, outputs
        torch.cuda.empty_cache()

        # Create results
        results = []
        for item, response in zip(batch_data, responses):
            results.append({
                "question": item['question'],
                "answer": item['answer'],
                "model_output": response
            })

        return results

    def process_dataset(self, input_path: str, output_path: str):
        """Process HotPot QA dataset using streaming and incremental saves."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            current_batch = []
            total_processed = 0
            
            # Process data in streaming fashion
            for item in tqdm(self.stream_json_data(input_path)):
                current_batch.append(item)
                
                if len(current_batch) >= BATCH_SIZE:
                    # Process batch
                    results = self.process_batch(current_batch)
                    
                    # Save results (first batch writes new file, subsequent batches append)
                    mode = 'w' if total_processed == 0 else 'a'
                    self.save_batch_results(results, output_path, mode)
                    
                    # Clear batch and increment counter
                    total_processed += len(current_batch)
                    current_batch = []
                    
                    # Memory cleanup
                    gc.collect()
            
            # Process any remaining items
            if current_batch:
                results = self.process_batch(current_batch)
                self.save_batch_results(results, output_path, 'a')
                total_processed += len(current_batch)
            
            self.logger.info(f"Processed {total_processed} examples. Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error processing dataset: {e}")
            raise

def main():
    processor = HotpotQAProcessor()
    processor.process_dataset(
        input_path="hotpot_training_data.json",
        output_path="hotpot_results.json"
    )

if __name__ == "__main__":
    main()