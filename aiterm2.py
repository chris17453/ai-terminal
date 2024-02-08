import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging

# Set transformers logging level to error
transformers_logging.set_verbosity_error()

class Terminal:
    def __init__(self, model_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype="auto").cuda()
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, code):
        prompt = f"You are a bash terminal. \
            I want you to only reply with the terminal output inside one unique code block, and nothing else. \
            Do not write explanations. Do not type commands unless I instruct you to do so. if there is an error, display an error message from bash. \
            please execute the following command. \n{code}"        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.3, do_sample=True)
        output_text = self.tokenizer.decode(tokens[0], skip_special_tokens=True).replace('"""', '').replace("'''", '')
        return output_text[len(self.tokenizer.decode(inputs['input_ids'][0])):]

    def run(self):
        print("AI-terminal is running. Type 'exit' to quit.")
        while True:
            user_input = input("[user@aiterminal]$ ")
            if user_input.lower() == 'exit':
                print("Exiting AI-terminal.")
                break
            print(self.generate(user_input))

def main():
    parser = argparse.ArgumentParser(description="Process the model directory.")
    parser.add_argument('--model-dir', type=str, required=True, help='Path to the model directory')
    args = parser.parse_args()
    if not os.path.isdir(args.model_dir):
        print(f"The model directory '{args.model_dir}' does not exist or is not a directory.")
        return
    Terminal(args.model_dir).run()

if __name__ == "__main__":
    main()