import tkinter as tk
import sys
import torch
from transformers import pipeline, AutoTokenizer, TextStreamer
import threading


class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Scroll to the end
        
    def flush(self):
        pass


class ChatApp:
    def __init__(self, master):
        self.master = master
        master.title("Chat Application")

        # Create a Text widget to display the conversation
        self.output_text = tk.Text(master, wrap=tk.WORD)
        self.output_text.pack(expand=True, fill=tk.BOTH)
        self.output_text.tag_config('user', foreground="blue", font=("Courier", "10", "bold"))
        self.output_text.tag_config('assistant', foreground="green", font=("Courier", "10", "bold"))
        
        self.pipe = pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            torch_dtype=torch.float16, 
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        self.streamer = TextStreamer(skip_prompt=True, tokenizer=self.tokenizer)

        self.system_instruction = {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate. Keep your response polite and concise.",
        }

        self.chat = []
        self.chat += [self.system_instruction]
        
        # Create an Entry widget for user input
        self.input_entry = tk.Entry(master)
        self.input_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Create a button to trigger the chat model
        self.chat_button = tk.Button(master, text="Chat", command=self.chatting)
        self.chat_button.pack(side=tk.RIGHT)

        # Redirect stdout to the Text widget
        sys.stdout = StdoutRedirector(self.output_text)

        print("#### Let's chat! ####")
    
    
    def chatting(self):
        # Get user input from the Entry widget
        user_input = self.input_entry.get()
        
        # Display user input in the conversation window
        self.update_text(f"\n[Me]: ", 'user')
        
        self.update_text(f"{user_input}\n")
        
        self.update_text(f"\n[ChatBot]: ", 'assistant')

        # Process user input using the chat model in a separate thread
        threading.Thread(target=self.process_user_input, args=(user_input,)).start()

        # Clear the Entry widget for the next input
        self.input_entry.delete(0, tk.END)

    def update_text(self, message, entity=None):
        # Update the Text widget in a thread-safe manner
        if entity:
            self.output_text.insert(tk.END, message, entity)
        else:
            self.output_text.insert(tk.END, message)
        self.output_text.see(tk.END)
    
    def process_user_input(self, user_input):
        user_input = {"role": "user", "content": user_input}
        self.chat += [user_input]
        formatted_chat = self.pipe.tokenizer.apply_chat_template(
            self.chat, tokenize=False, add_generation_prompt=True)
        
        # print("\nChatBot: ", end="")
        outputs = self.pipe(
            formatted_chat, streamer=self.streamer,
            max_new_tokens=512, do_sample=True, temperature=0.7, top_k=50, top_p=0.95,
            pad_token_id=self.tokenizer.eos_token_id,
            stop_sequence='</s>',
        )
        
        self.chat += [{"role": "assistant", "content": outputs[0]["generated_text"][len(formatted_chat):]}]
        if len(self.chat) > 3:
            self.chat = self.chat[:1] + self.chat[-2:]

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.bind('<Return>', lambda event=None: app.chat_button.invoke())
    root.mainloop()