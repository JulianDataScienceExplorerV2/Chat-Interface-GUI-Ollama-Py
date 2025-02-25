# Chat Interface GUI with Ollama (Python)

A Python-based graphical interface for interacting with **Ollama** language models. This application allows you to select a model, input prompts, and receive real-time responses while maintaining a conversation history for better context.


![image](https://github.com/user-attachments/assets/170b7285-136a-44e5-b29a-77534f0f603e)

---

## Features

- **Select a Model**: Choose from a list of available Ollama models to interact with. The dropdown menu allows you to easily switch between different models.
- **Generate Responses**: Input your prompt in the text box and click "Generate Response" to receive real-time answers from the selected model.
- **Export Responses**: Save the generated responses to a text file for future reference or documentation.
- **No Docker Required**: The application is designed to run directly on your local machine without the need for Docker or complex setups.
- **User-Friendly Interface**: The intuitive and modern interface makes it easy for users to interact with the models and manage conversations.
- **Syntax Highlighting**: Responses containing Python code are highlighted for better readability.

---

## Technologies Used

- **Python**: Core programming language.
- **Tkinter**: For building the graphical user interface (GUI).
- **Ollama**: Local language models for generating responses.
- **LangGraph**: For managing conversation history and state.
- **Pygments**: For syntax highlighting in responses.
- **Requests**: For fetching available Ollama models.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Chat-Interface-GUI-Ollama-Py.git
   cd Chat-Interface-GUI-Ollama-Py

Ensure you have Python installed, then run:
   pip install -r requirements.txt

Run the Application:
   python main.py
