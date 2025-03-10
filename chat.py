import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(prompt, model, tokenizer, device):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=512,  # Adjust the max_length as needed
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    parser = argparse.ArgumentParser(description="Chat with your fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    args = parser.parse_args()

    model_path = args.model_path
    print(f"Evaluating model {model_path}")

    # Load the fine-tuned model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    print("Chat with your fine-tuned model. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = generate_response(user_input, model, tokenizer, device)
        print(f"Model: {response}")

if __name__ == "__main__":
    main()