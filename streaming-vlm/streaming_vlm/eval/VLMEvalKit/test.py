
from vlmeval.config import supported_VLM
model = supported_VLM['Qwen2.5-VL-7B-Instruct']()
# Upstream implementation note.
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # Upstream implementation note.
# Upstream implementation note.
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # Upstream implementation note.
