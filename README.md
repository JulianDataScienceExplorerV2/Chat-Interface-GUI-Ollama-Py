# Silane Chat Interface

A Python-based graphical interface for interacting with **Ollama** language models. This application allows you to select a model, input prompts, and receive real-time responses while maintaining a conversation history for better context.

---

## Features

- **Modern Graphical Interface**: Dark-themed and user-friendly design built with `tkinter`.
- **Ollama Integration**: Direct connection to local Ollama models.
- **Conversation History**: Uses **LangGraph** to maintain context across interactions.
- **Syntax Highlighting**: Python code highlighting in responses using `Pygments`.
- **Response Export**: Save responses to a text file for later use.
- **Visual Indicators**: Animations and clear messages to enhance the user experience.

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
   git clone https://github.com/your-username/silane-chat-interface.git
   cd silane-chat-interface
