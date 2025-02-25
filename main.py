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

# Definición del estado con persistencia
class ChatState(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]  # Reducer para agregar mensajes

class OllamaInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Silane Model Interface")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2E3440")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.modelos = self.obtener_modelos()
        if not self.modelos:
            messagebox.showerror("Error", "No se pudieron cargar los modelos. Verifica la conexión con Ollama.")
            self.root.destroy()
            return

        self.llm = None
        self.checkpointer = MemorySaver()  # Checkpointer para persistencia
        self.grafo_conversacion = self.crear_grafo_conversacion()  # Grafo de LangGraph
        self.respuesta_queue = queue.Queue()
        self.verificar_respuesta()
        self.animacion_activa = False
        self.current_thread = None  # Hilo actual de conversación

        self.crear_interfaz()

    def obtener_modelos(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return [modelo["name"] for modelo in response.json()["models"]]
            else:
                messagebox.showerror("Error", f"No se pudieron obtener los modelos: {response.text}")
                return []
        except Exception as e:
            messagebox.showerror("Error", f"Error de conexión: {str(e)}")
            return []

    def crear_interfaz(self):
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

        pregunta_frame = ttk.Frame(main_frame)
        pregunta_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        pregunta_frame.grid_columnconfigure(0, weight=1)
        pregunta_frame.grid_columnconfigure(1, weight=1)
        pregunta_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(pregunta_frame, text="Selecciona un modelo:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.combo_modelos = ttk.Combobox(pregunta_frame, values=self.modelos, width=40)
        self.combo_modelos.grid(row=0, column=1, pady=5, padx=5, sticky="ew")
        self.combo_modelos.current(0)

        ttk.Label(pregunta_frame, text="Ingresa tu prompt:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        self.entrada_prompt = tk.Text(pregunta_frame, height=10, width=50, bg="#3B4252", fg="#ECEFF4", insertbackground="#ECEFF4", font=("Arial", 14))
        self.entrada_prompt.grid(row=1, column=1, pady=5, padx=5, sticky="nsew")
        self.entrada_prompt.bind("<Control-Return>", self.iniciar_generacion_respuesta)

        self.boton_generar = ttk.Button(pregunta_frame, text="Generar Respuesta", command=self.iniciar_generacion_respuesta)
        self.boton_generar.grid(row=2, column=1, pady=10, padx=5, sticky="e")

        respuesta_frame = ttk.Frame(main_frame)
        respuesta_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        respuesta_frame.grid_columnconfigure(0, weight=1)
        respuesta_frame.grid_rowconfigure(1, weight=1)

        ttk.Label(respuesta_frame, text="Respuesta del Modelo:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        self.area_respuesta = scrolledtext.ScrolledText(respuesta_frame, wrap=tk.WORD, width=70, height=20, state=tk.DISABLED, bg="#3B4252", fg="#ECEFF4", insertbackground="#ECEFF4", font=("Arial", 14))
        self.area_respuesta.grid(row=1, column=0, pady=5, padx=5, sticky="nsew")
        self.area_respuesta.config(state=tk.NORMAL)
        self.area_respuesta.insert(tk.END, "Por favor, selecciona un modelo y escribe un prompt para comenzar.")
        self.area_respuesta.config(state=tk.DISABLED)

        self.boton_exportar = ttk.Button(respuesta_frame, text="Exportar Respuesta", command=self.exportar_respuesta, state=tk.DISABLED)
        self.boton_exportar.grid(row=2, column=0, pady=10, padx=5, sticky="e")

        self.estado_label = ttk.Label(pregunta_frame, text="", font=("Arial", 12))
        self.estado_label.grid(row=4, column=1, pady=5, padx=5, sticky="w")

    def crear_grafo_conversacion(self):
        # Crear el grafo de LangGraph
        builder = StateGraph(ChatState)

        # Nodo para generar una respuesta usando el modelo
        builder.add_node("chatbot", self.invocar_modelo)
        builder.add_edge(START, "chatbot")  # Punto de inicio
        builder.add_edge("chatbot", END)    # Punto de finalización

        # Compilar el grafo con persistencia
        return builder.compile(checkpointer=self.checkpointer)

    def invocar_modelo(self, state: ChatState):
        # Generar una respuesta usando el modelo
        respuesta = self.llm.invoke(state["messages"])
        return {"messages": [AIMessage(content=respuesta)]}

    def iniciar_generacion_respuesta(self, event=None):
        # Deshabilitar el botón mientras se genera la respuesta
        self.boton_generar.config(state=tk.DISABLED)
        self.boton_exportar.config(state=tk.DISABLED)
        self.area_respuesta.config(state=tk.NORMAL)
        self.area_respuesta.delete("1.0", tk.END)
        self.area_respuesta.config(state=tk.DISABLED)

        modelo_seleccionado = self.combo_modelos.get()
        prompt = self.entrada_prompt.get("1.0", tk.END).strip()

        if not modelo_seleccionado or not prompt:
            messagebox.showwarning("Advertencia", "Selecciona un modelo y escribe un prompt.")
            self.boton_generar.config(state=tk.NORMAL)
            self.estado_label.config(text="")
            return

        self.llm = OllamaLLM(model=modelo_seleccionado)

        # Crear un nuevo thread si es necesario
        if not self.current_thread:
            self.current_thread = f"thread_{time.time()}"

        # Ejecutar en un hilo separado
        threading.Thread(
            target=self.generar_respuesta,
            args=(prompt, self.current_thread),
            daemon=True
        ).start()

    def generar_respuesta(self, prompt, thread_id):
        try:
            # Configuración del thread para persistencia
            config = {"configurable": {"thread_id": thread_id}}

            # Crear el mensaje inicial
            mensaje_inicial = HumanMessage(content=prompt)

            # Ejecutar el grafo con persistencia
            for event in self.grafo_conversacion.stream(
                {"messages": [mensaje_inicial]},
                config=config,
                stream_mode="values"
            ):
                if "messages" in event:
                    respuesta = event["messages"][-1].content
                    self.respuesta_queue.put(respuesta)

        except Exception as e:
            self.respuesta_queue.put(f"Error: {str(e)}")
        finally:
            self.animacion_activa = False

    def verificar_respuesta(self):
        try:
            respuesta = self.respuesta_queue.get_nowait()
            self.mostrar_respuesta(respuesta)
            self.boton_generar.config(state=tk.NORMAL)  # Habilitar el botón
            self.boton_exportar.config(state=tk.NORMAL)  # Habilitar el botón de exportar
            self.estado_label.config(text="")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.verificar_respuesta)

    def mostrar_respuesta(self, respuesta):
        self.area_respuesta.config(state=tk.NORMAL)
        self.area_respuesta.delete("1.0", tk.END)

        respuesta_formateada = respuesta

        if "python" in self.combo_modelos.get().lower():
            lexer = PythonLexer()
            tokens = lex(respuesta_formateada, lexer)
            for token_type, value in tokens:
                self.area_respuesta.insert(tk.END, value, token_type)
        else:
            self.area_respuesta.insert(tk.END, respuesta_formateada)

        self.area_respuesta.config(state=tk.DISABLED)

    def exportar_respuesta(self):
        respuesta = self.area_respuesta.get("1.0", tk.END).strip()
        if not respuesta:
            messagebox.showwarning("Advertencia", "No hay respuesta para exportar.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Archivos de texto", "*.txt")])
        if file_path:
            try:
                with open(file_path, "w") as file:
                    file.write(respuesta)
                messagebox.showinfo("Éxito", "Respuesta exportada correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo exportar la respuesta: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = OllamaInterface(root)
    root.mainloop()
