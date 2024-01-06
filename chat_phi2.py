import torch
from transformers import pipeline, AutoTokenizer, TextStreamer

pipe = pipeline(
    "text-generation", 
    model="cognitivecomputations/dolphin-2_6-phi-2", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("cognitivecomputations/dolphin-2_6-phi-2", trust_remote_code=True)

streamer = TextStreamer(skip_prompt=True, tokenizer=tokenizer)

system_instruction = {
    "role": "system",
    "content": "You are Dolphin, a helpful AI assistant.",
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
        max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id,
        stop_sequence='<|im_end|>',
    )
    
    chat += [{"role": "assistant", "content": outputs[0]["generated_text"][len(formatted_chat):]}]
    if len(chat) > 3:
       chat = chat[:1] + chat[-2:]
    
    
    