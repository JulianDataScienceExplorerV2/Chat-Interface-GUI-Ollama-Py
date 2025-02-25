import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import time
from pygments import lex
from pygments.lexers import PythonLexer
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
import requests

# State definition with persistence
class ChatState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # Reducer to append messages

class OllamaInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Chat Interface GUI with Ollama")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2E3440")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.models = self.get_models()
        if not self.models:
            messagebox.showerror("Error", "Could not load models. Please check your connection to Ollama.")
            self.root.destroy()
            return

        self.llm = None
        self.checkpointer = MemorySaver()  # Checkpointer for persistence
        self.conversation_graph = self.create_conversation_graph()  # LangGraph graph
        self.response_queue = queue.Queue()
        self.animation_active = False
        self.current_thread = None  # Current conversation thread
        self.window_open = True  # Flag to check if the window is open

        self.create_interface()
        self.check_response()

    def get_models(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return [model["name"] for model in response.json()["models"]]
            else:
                messagebox.showerror("Error", f"Could not fetch models: {response.text}")
                return []
        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {str(e)}")
            return []

    def create_interface(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        style = ttk.Style()
        style.configure("TLabel", foreground="#ECEFF4", background="#2E3440", font=("Arial", 12))
        style.configure("TButton", foreground="#ECEFF4", background="#4C566A", font=("Arial", 12))
        style.configure("TCombobox", fieldbackground="#3B4252", foreground="#ECEFF4", font=("Arial", 12))
        style.configure("TFrame", background="#2E3440")
        style.configure("TText", background="#3B4252", foreground="#ECEFF4", font=("Arial", 14))

        prompt_frame = ttk.Frame(main_frame)
        prompt_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        prompt_frame.grid_columnconfigure(0, weight=1)
        prompt_frame.grid_columnconfigure(1, weight=1)
        prompt_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(prompt_frame, text="Select a model:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.model_combobox = ttk.Combobox(prompt_frame, values=self.models, width=40)
        self.model_combobox.grid(row=0, column=1, pady=5, padx=5, sticky="ew")
        self.model_combobox.current(0)

        ttk.Label(prompt_frame, text="Enter your prompt:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.prompt_entry = tk.Text(prompt_frame, height=10, width=50, bg="#3B4252", fg="#ECEFF4", insertbackground="#ECEFF4", font=("Arial", 14))
        self.prompt_entry.grid(row=1, column=1, pady=5, padx=5, sticky="nsew")
        self.prompt_entry.bind("<Control-Return>", self.start_response_generation)

        self.generate_button = ttk.Button(prompt_frame, text="Generate Response", command=self.start_response_generation)
        self.generate_button.grid(row=2, column=1, pady=10, padx=5, sticky="e")

        response_frame = ttk.Frame(main_frame)
        response_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        response_frame.grid_columnconfigure(0, weight=1)
        response_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(response_frame, text="Model Response:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.response_area = scrolledtext.ScrolledText(response_frame, wrap=tk.WORD, width=70, height=20, state=tk.DISABLED, bg="#3B4252", fg="#ECEFF4", insertbackground="#ECEFF4", font=("Arial", 14))
        self.response_area.grid(row=1, column=0, pady=5, padx=5, sticky="nsew")
        self.response_area.config(state=tk.NORMAL)
        self.response_area.insert(tk.END, "Please select a model and enter a prompt to begin.")
        self.response_area.config(state=tk.DISABLED)

        self.export_button = ttk.Button(response_frame, text="Export Response", command=self.export_response, state=tk.DISABLED)
        self.export_button.grid(row=2, column=0, pady=10, padx=5, sticky="e")

        self.status_label = ttk.Label(prompt_frame, text="", font=("Arial", 12))
        self.status_label.grid(row=4, column=1, pady=5, padx=5, sticky="w")

    def create_conversation_graph(self):
        # Create the LangGraph graph
        builder = StateGraph(ChatState)

        # Node to generate a response using the model
        builder.add_node("chatbot", self.invoke_model)
        builder.add_edge(START, "chatbot")  # Entry point
        builder.add_edge("chatbot", END)    # Exit point

        # Compile the graph with persistence
        return builder.compile(checkpointer=self.checkpointer)

    def invoke_model(self, state: ChatState):
        # Generate a response using the model
        response = self.llm.invoke(state["messages"])
        return {"messages": [AIMessage(content=response)]}

    def start_response_generation(self, event=None):
        # Disable the button while generating the response
        self.generate_button.config(state=tk.DISABLED)
        self.export_button.config(state=tk.DISABLED)
        self.response_area.config(state=tk.NORMAL)
        self.response_area.delete("1.0", tk.END)
        self.response_area.insert(tk.END, "Thinking...")  # Show "Thinking..." while generating the response
        self.response_area.config(state=tk.DISABLED)

        selected_model = self.model_combobox.get()
        prompt = self.prompt_entry.get("1.0", tk.END).strip()

        if not selected_model or not prompt:
            messagebox.showwarning("Warning", "Please select a model and enter a prompt.")
            self.generate_button.config(state=tk.NORMAL)
            self.status_label.config(text="")
            return

        self.llm = OllamaLLM(model=selected_model)

        # Create a new thread if necessary
        if not self.current_thread:
            self.current_thread = f"thread_{time.time()}"

        # Run in a separate thread
        threading.Thread(
            target=self.generate_response,
            args=(prompt, self.current_thread),
            daemon=True
        ).start()

    def generate_response(self, prompt, thread_id):
        try:
            # Thread configuration for persistence
            config = {"configurable": {"thread_id": thread_id}}

            # Create the initial message
            initial_message = HumanMessage(content=prompt)

            # Run the graph with persistence
            for event in self.conversation_graph.stream(
                {"messages": [initial_message]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in event:
                    response = event["messages"][-1].content
                    self.response_queue.put(response)  # Only put the response in the queue

        except Exception as e:
            self.response_queue.put(f"Error: {str(e)}")
        finally:
            self.animation_active = False

    def check_response(self):
        if not self.window_open:
            return  # Stop if the window is closed

        try:
            response = self.response_queue.get_nowait()
            self.show_response(response)
            self.generate_button.config(state=tk.NORMAL)  # Enable the button
            self.export_button.config(state=tk.NORMAL)  # Enable the export button
            self.status_label.config(text="")
        except queue.Empty:
            pass
        finally:
            if self.window_open:
                self.root.after(100, self.check_response)

    def show_response(self, response):
        self.response_area.config(state=tk.NORMAL)
        self.response_area.delete("1.0", tk.END)  # Clear the "Thinking..." message

        formatted_response = response

        if "python" in self.model_combobox.get().lower():
            lexer = PythonLexer()
            tokens = lex(formatted_response, lexer)
            for token_type, value in tokens:
                self.response_area.insert(tk.END, value, token_type)
        else:
            self.response_area.insert(tk.END, formatted_response)  # Insert only the model's response

        self.response_area.config(state=tk.DISABLED)

    def export_response(self):
        response = self.response_area.get("1.0", tk.END).strip()
        if not response:
            messagebox.showwarning("Warning", "No response to export.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "w") as file:
                    file.write(response)
                messagebox.showinfo("Success", "Response exported successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export response: {str(e)}")

    def on_close(self):
        self.window_open = False  # Set the flag to False when the window is closed
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaInterface(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)  # Handle window close event
    root.mainloop()
