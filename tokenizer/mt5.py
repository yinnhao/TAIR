from transformers import MT5Tokenizer, MT5ForConditionalGeneration
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")


# text = "你好，世界！这是一段中文文本。"
# tokens = tokenizer.tokenize(text)
# print(tokens)

input_ids = tokenizer("translate chinese to German: 你好", return_tensors="pt").input_ids
print(input_ids)
print(tokenizer.decode(input_ids[0]))