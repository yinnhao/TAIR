from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_ids = tokenizer("translate chinese to German: 你好", return_tensors="pt").input_ids
print(input_ids)
print(tokenizer.decode(input_ids[0]))
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Das Haus ist wunderbar.