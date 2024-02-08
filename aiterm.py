import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from transformers import logging as transformers_logging

# Set the transformers library's logging level to error only
transformers_logging.set_verbosity_error()


class terminal:
    def __init__(self,model_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True,  torch_dtype="auto",)
        self.model.cuda()
                # Explicitly set pad_token_id if it's None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, code, attempts=3):
        # Required sections

        # Prepare the prompt
        prompt = f"You are a bash terminal. \
            I want you to only reply with the terminal output inside one unique code block, and nothing else. \
            Do not write explanations. Do not type commands unless I instruct you to do so. if there is an error, display an error message from bash. \
            please execute the following command. \n{code}"

        inputs = self.tokenizer(prompt, return_token_type_ids=False, return_tensors="pt").to("cuda")
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
        )
        generated_text = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        output_text = generated_text[len(self.tokenizer.decode(inputs['input_ids'][0])):]

        output_text=output_text.replace('\"\"\"','')
        output_text=output_text.replace("\'\'\'",'')
        
        return output_text



    def run(self):
        print("ai-terminal is running. Type 'exit' to quit.")
        while True:
            # Taking input from the user
            user_input = input("[user@aitermal]$")
            
            if user_input.lower() == 'exit':
                print("Exiting ai-terminal.")
                break
            
            # Sending the user input to another component and getting a response
            response = self.generate(user_input)
            
            # Displaying the response in the terminal
            print(response)

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the model directory.")
    parser.add_argument('--model-dir', type=str, help='Path to the model directory')
    args = parser.parse_args()
    
     # Assuming you want to check if the model_dir exists and print a message
    if os.path.isdir(args.model_dir)!=True:
        print(f"The model directory '{args.model_dir}' does not exist or is not a directory.")

    cli = terminal(args.model_dir)
    cli.run()

# To run the CLI
if __name__ == "__main__":
    main()