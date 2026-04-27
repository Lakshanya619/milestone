import customtkinter as ctk
from threading import Thread
from openai import OpenAI
from huggingface_hub import InferenceClient
import config

# -------- SETTINGS --------
GROQ_URL = "https://api.groq.com/openai/v1"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# -------- MEMORY --------
chat_history = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Remember previous conversation and give clear, structured answers."
    }
]

# -------- AI FUNCTIONS --------

def get_groq_response(prompt):
    try:
        client = OpenAI(api_key=config.GROQ_API_KEY, base_url=GROQ_URL)

        chat_history.append({"role": "user", "content": prompt})

        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=chat_history,
            temperature=0.5,
        )

        reply = r.choices[0].message.content

        chat_history.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        return f"Groq Error: {e}"


def get_hf_response(prompt):
    try:
        client = InferenceClient(
            model="meta-llama/Llama-3.1-8B-Instruct",
            token=config.HF_API_KEY
        )

        chat_history.append({"role": "user", "content": prompt})

        r = client.chat_completion(
            messages=chat_history
        )

        reply = r.choices[0].message.content

        chat_history.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        return f"HF Error: {e}"


def generate_response(prompt, provider):
    if provider == "hf":
        return get_hf_response(prompt)
    return get_groq_response(prompt)


# -------- APP --------
class CobyAI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Coby AI 🤖")
        self.geometry("900x650")

        # Top bar
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=10, pady=5)

        self.provider = ctk.StringVar(value="groq")

        ctk.CTkLabel(top, text="Coby AI", font=("Segoe UI", 16, "bold")).pack(side="left")

        ctk.CTkOptionMenu(
            top,
            values=["groq", "hf"],
            variable=self.provider,
            width=100
        ).pack(side="right")

        # Chat area
        self.chat = ctk.CTkTextbox(self, font=("Segoe UI", 13))
        self.chat.pack(fill="both", expand=True, padx=10, pady=10)

        # Input area
        bottom = ctk.CTkFrame(self)
        bottom.pack(fill="x", padx=10, pady=10)

        self.input_box = ctk.CTkTextbox(bottom, height=100, font=("Segoe UI", 12))
        self.input_box.pack(side="left", fill="x", expand=True, padx=(0, 10))

        send_btn = ctk.CTkButton(bottom, text="Send", width=80, command=self.send)
        send_btn.pack(side="right")

        # Enter handling
        self.input_box.bind("<Return>", self.handle_enter)

    def handle_enter(self, event):
        if not event.state & 0x0001:  # Shift not pressed
            self.send()
            return "break"

    def send(self):
        prompt = self.input_box.get("1.0", "end").strip()
        if not prompt:
            return

        self.chat.insert("end", f"\n👤 You:\n{prompt}\n\n")
        self.input_box.delete("1.0", "end")

        self.chat.insert("end", "🤖 Thinking...\n\n")
        self.chat.see("end")

        def run():
            response = generate_response(prompt, self.provider.get())

            # Remove last "Thinking..."
            content = self.chat.get("1.0", "end")
            content = content.rsplit("🤖 Thinking...", 1)[0]

            self.chat.delete("1.0", "end")
            self.chat.insert("end", content)

            self.chat.insert("end", f"🤖 Coby:\n{response}\n\n")
            self.chat.see("end")

        Thread(target=run).start()


# -------- RUN --------
app = CobyAI()
app.mainloop()