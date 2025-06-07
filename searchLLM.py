from transformers import AutoTokenizer, AutoModelForCausalLM

# Choose a model
#model_name = "gpt2" # Basic Model
#model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_name = "microsoft/phi-3-mini-4k-instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ask the user for a prompt
user_prompt = input("Enter your prompt: ")

# Tokenize the input
inputs = tokenizer(user_prompt, return_tensors="pt")

# Generate output
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

# Decode and print the result
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Output:")
print(result)
