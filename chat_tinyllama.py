import torch
from transformers import pipeline, AutoTokenizer, TextStreamer

pipe = pipeline(
    "text-generation", 
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    torch_dtype=torch.float16, 
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

streamer = TextStreamer(skip_prompt=True, tokenizer=tokenizer)

system_instruction = {
    "role": "system",
    "content": "You are a friendly chatbot who always responds in the style of a pirate.",
}

chat = []
chat += [system_instruction]

print("#### Let's chat! ####")

while True:
    print("\nMe: ", end="")
    user_input = input("")
    user_input = {"role": "user", "content": user_input}
    chat += [user_input]
    formatted_chat = pipe.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True)
    
    print("\nChatBot: ", end="")
    outputs = pipe(
        formatted_chat, streamer=streamer,
        max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95,
    )
    
    chat += [{"role": "assistant", "content": outputs[0]["generated_text"][len(formatted_chat):]}]
    if len(chat) > 3:
       chat = chat[:1] + chat[-2:]
    
    
    