
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("tokenizer pad_token_id: ", tokenizer.pad_token_id)
print("tokenizer pad_token_type_id: ", tokenizer.pad_token_type_id)
print("tokenizer eos_token_id: ", tokenizer.eos_token_id)