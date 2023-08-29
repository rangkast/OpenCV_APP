import openai
import tkinter as tk
from tkinter import ttk

# OpenAI API key 설정
api_key = "your-api-key-here"

def chat_with_gpt4(prompt):
    openai.api_key = api_key
    model_engine = "text-davinci-002"  # 예시 엔진 이름입니다; 실제 엔진 이름은 문서를 참조하세요.
    max_tokens = 100  # 응답의 최대 토큰 수
    
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens
    )
    
    message = response.choices[0].text.strip()
    return message

def send_message():
    user_message = user_input.get()
    conversation["state"] += f"You: {user_message}\n"
    prompt = conversation["state"] + "ChatGPT:"
    
    gpt_response = chat_with_gpt4(prompt)
    
    conversation["state"] += f"ChatGPT: {gpt_response}\n"
    
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {user_message}\n")
    chat_history.insert(tk.END, f"ChatGPT: {gpt_response}\n")
    chat_history.config(state=tk.DISABLED)
    
    user_input.delete(0, tk.END)

# Tkinter GUI 설정
root = tk.Tk()
root.title("Chat with GPT-4")

conversation = {"state": "The following is a conversation with ChatGPT:\n"}

chat_frame = ttk.Frame(root, padding="10")
chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

chat_history = tk.Text(chat_frame, wrap=tk.WORD, width=50, height=15)
chat_history.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
chat_history.config(state=tk.DISABLED)

user_input = ttk.Entry(chat_frame, width=50)
user_input.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

send_button = ttk.Button(chat_frame, text="Send", command=send_message)
send_button.grid(row=1, column=1)

root.mainloop()
