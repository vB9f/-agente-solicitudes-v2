import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
import time
from sqlalchemy import text

# Importar herramientas
from tools.registrar_solicitud import create_tool_registrar_solicitud
from tools.consultar_estado import create_tool_consultar_estado
from tools.actualizar_solicitud import create_tool_actualizar_solicitud

from tools.busqueda_documental import create_tool_busqueda_documental

# --- CONFIGURACIÓN DE PÁGINA Y LLM ---
st.set_page_config(page_title="Agente de Reembolsos Médicos", layout="wide")

# Credenciales de OpenAI
with open("openai.txt") as archivo:
    os.environ["OPENAI_API_KEY"] = archivo.read().strip()
    
# Credenciales de PostgreSQL
with open("postgresql.txt") as archivo:
    uribd = archivo.read()
    
# Credenciales de Langsmith
with open("langchain.txt") as archivo:
    os.environ["LANGCHAIN_API_KEY"] = archivo.read().strip()
    os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" # Seguimiento
    os.environ["LANGCHAIN_PROJECT"] = "S09-eiagurp" # Nombre del proyecto
    
# Credenciales de Elasticsearch
with open("elasticstore.txt") as archivo:
    key_elastic = archivo.read().strip()
    ELASTIC_URL = "http://XX.XX.XX.XX:9200" # Ingresar IP pública del vector store
    ELASTIC_USER = "elastic"
    ELASTIC_INDEX = "AAAAAAA" # Ingresar nombre del index

# Iniciar conexión a base de datos SQL
try:
    DB_CONN = SQLDatabase.from_uri(uribd)
except Exception as e:
    st.error(f"Error al conectar con la base de datos: {e}")
    DB_CONN = None

# Iniciar modelo y memoria
@st.cache_resource
def setup_agent_llm():
    """Configura el modelo LLM y el checkpointer para el agente."""
    model = ChatOpenAI(model="gpt-4o-mini")
    memory = MemorySaver()
    return model, memory

model, memory = setup_agent_llm()

# Iniciar vector store
@st.cache_resource
def setup_vector_store():
    """Configura los embeddings y la conexión al vector store."""
    try:
        embedding = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = ElasticsearchStore(
            es_url=ELASTIC_URL,
            es_user=ELASTIC_USER,
            es_password=key_elastic,
            index_name=ELASTIC_INDEX,
            embedding=embedding
        )
        return vector_store
    except Exception as e:
        st.error(f"Error al conectar con Elasticsearch: {e}")
        return None

VECTOR_STORE = setup_vector_store()

# Estado del grafo
class AgenteState(TypedDict):
    """Representa el estado del grafo."""
    messages: Annotated[List[HumanMessage], operator.add]
    next: str

# Nodo de agente (Función auxiliar para ejecutar cualquier agente)
def agent_node(state, agent, name):
    result = agent.invoke(state)
    # El agente debe devolver un mensaje que se añade al historial
    return {"messages": [result["messages"][-1]]}

# Nodo supervisor (define el flujo)
def supervisor_node(state: AgenteState):
    """
    Decide qué agente debe ser el siguiente en responder.
    - 'documentacion' si es una pregunta teórica (Cuál es el proceso, cómo funciona, requisitos, etc.).
    - 'usuario_externo' si es una solicitud de acción (registrar, consultar, actualizar).
    - 'FIN' si ya se respondió o la consulta es irrelevante.
    """
    user_query = state["messages"][-1].content
    
    # Modelo LLM para la toma de decisiones
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Eres un agente supervisor de un sistema de reembolsos. Tu tarea es enrutar la consulta del usuario. "
         "Basado en la última consulta, decide a qué equipo debe ir: "
         "- **DOCUMENTACION**: Si la pregunta es sobre políticas, procedimientos, requisitos, qué documentos llevar, o cualquier información teórica general. "
         "- **USUARIO_EXTERNO**: Si la pregunta implica una ACCIÓN sobre una solicitud: registrar una solicitud, consultar el estado o actualizar una solicitud. "
         "Tu respuesta DEBE ser una de las siguientes palabras ÚNICAMENTE: DOCUMENTACION, USUARIO_EXTERNO."
        ),
        ("human", f"Última consulta del usuario: {user_query}")
    ])
    
    cadena_decision = decision_prompt | model
    decision = cadena_decision.invoke({"user_query": user_query}).content.strip().upper()
    
    if "DOCUMENTACION" in decision:
        return {"next": "documentacion"}
    elif "USUARIO_EXTERNO" in decision:
        return {"next": "usuario_externo"}
    else:
        return {"next": "usuario_externo"}

# Generar instancias de herramientas
if DB_CONN is not None:
    registrar_solicitud_tool = create_tool_registrar_solicitud(DB_CONN)
    consultar_estado_tool = create_tool_consultar_estado(DB_CONN)
    actualizar_estado_tool = create_tool_actualizar_solicitud(DB_CONN)
else:
    registrar_solicitud_tool = None
    consultar_estado_tool = None
    actualizar_estado_tool = None

if VECTOR_STORE is not None:
    busqueda_documental_tool = create_tool_busqueda_documental(VECTOR_STORE)
else:
    busqueda_documental_tool = None

# --- AGENTES ---
def create_agent_for_role(rol):
    """Crea el agente de LangGraph con el conjunto de herramientas apropiado según el rol."""
    
    # Capturamos el usuario
    usuario_logueado_user = st.session_state.get('username', 'Usuario desconocido')
    usuario_logueado_fullname = st.session_state.get('display_name', 'Usuario desconocido')
    
    if rol == "Administrador":
        toolkit = [registrar_solicitud_tool, consultar_estado_tool, actualizar_estado_tool]
        st.session_state.rol_info = f"🩺 Usuario **Administrador** {usuario_logueado_fullname} (Acceso total)"
        prompt_instruccion = (
            f"El usuario logueado es **{usuario_logueado_user}** y tiene acceso total. "
            "Cuando uses la herramienta 'consultar_estado_tool', **NO incluyas el argumento 'usuario'** en la llamada. "
            f"Para la herramienta 'registrar_solicitud_tool', utiliza **{usuario_logueado_user}** y **{usuario_logueado_fullname}** automáticamente para los argumentos: 'usuario' y 'nombre_asegurado', respectivamente."
        )
    elif rol == "General":
        toolkit = [registrar_solicitud_tool, consultar_estado_tool]
        st.session_state.rol_info = f"👤 Usuario **General** {usuario_logueado_fullname} (Solo registro y consulta)"
        prompt_instruccion = (
            f"El usuario logueado es **{usuario_logueado_user}**. "
            f"Para las herramientas 'registrar_solicitud_tool' y 'consultar_estado_tool', utiliza **{usuario_logueado_user}** y **{usuario_logueado_fullname}** automáticamente para los argumentos: 'usuario' y 'nombre_asegurado', respectivamente. "
            "Solo puedes consultar solicitudes asociadas a tu usuario."
        )
    else:
        toolkit = [] # Sin herramientas para otros roles
        st.session_state.rol_info = "Rol desconocido."

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         f"Eres el agente de soporte para reembolsos médicos. {prompt_instruccion} "
         "Si el usuario menciona un 'Beneficiario', úsalo para el argumento 'nombre_beneficiario'. "
         "Céntrate siempre en responder únicamente a la última pregunta del usuario. NO repitas ni resumas acciones o confirmaciones de solicitudes ya procesadas en turnos anteriores de la conversación si es que el usuario no te las pide."
         "Usa las herramientas disponibles solo cuando sea necesario y sé cortés."),
        ("human", "{messages}")
    ])

    agent_instance = create_react_agent(model, toolkit, checkpointer=memory, prompt=prompt)
    return agent_instance

def create_documentacion_agent():
    toolkit = [busqueda_documental_tool]
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
          "Eres el agente de documentación. Tu ÚNICA función es usar la herramienta 'busqueda_documental' para encontrar el procedimiento o pasos de reembolsos en la base de datos vectorial y resumir la información encontrada. "
          "Si la herramienta no devuelve información, indica al usuario que no encontraste ese detalle."
        ),
        ("human", "{messages}")
    ])
    
    agent_instance = create_react_agent(model, toolkit, checkpointer=memory, prompt=prompt)
    return agent_instance

# --- GRAFO ---
def build_agent_graph(agente_usuario, agente_documentacion):
    """Construye el grafo supervisor que conecta los agentes de rol."""
    workflow = StateGraph(AgenteState)
    
    # Nodos de agentes
    workflow.add_node("documentacion", lambda state: agent_node(state, agente_documentacion, "documentacion"))
    workflow.add_node("usuario_externo", lambda state: agent_node(state, agente_usuario, "usuario_externo"))
    
    # Nodo supervisor
    workflow.add_node("supervisor", supervisor_node)
    
    # Flujo
    workflow.set_entry_point("supervisor")
    
    # El supervisor dirige a uno de los agentes
    workflow.add_conditional_edges(
        "supervisor", 
        lambda x: x["next"],
        {
            "documentacion": "documentacion",
            "usuario_externo": "usuario_externo"
        }
    )
    
    # Después de que un agente responde, el flujo termina
    workflow.add_edge("documentacion", END)
    workflow.add_edge("usuario_externo", END)

    # Compilar el grafo con la memoria/checkpointer
    app = workflow.compile(checkpointer=memory)
    return app

# --- VISTAS DE STREAMLIT ---
def login_view():
    """Muestra el formulario de login y maneja la autenticación."""
    st.title("Sistema de Reembolsos Médicos")
    st.header("Inicio de Sesión")

    if st.session_state.login_attempt_successful:
        st.info("Iniciando sesión...")
        return

    with st.form("login_form"):
        usuario = st.text_input("Usuario")
        contrasena = st.text_input("Contraseña", type="password")
        boton_login = st.form_submit_button("Ingresar")

        if boton_login:
            if DB_CONN is None:
                st.error("No se puede iniciar sesión, la conexión no está disponible.")
                return
            
            sql_query = text("""
            SELECT 
                "usuario", 
                "tipousuario", 
                CONCAT("nombres", ' ', "apellidopaterno", ' ', "apellidomaterno") AS NombreCompleto
            FROM usuarios_sistema 
            WHERE 
                "usuario" = :user AND 
                "contrasena" = :password AND 
                "estado" = 'Activo';
            """)
            
            try:
                user_data = None
                                
                params = {"user": usuario, "password": contrasena}
            
                with DB_CONN._engine.connect() as conexion:
                    resultado = conexion.execute(sql_query, params)
                    user_data = resultado.fetchone()
            
                if user_data:                
                    user = user_data[0]
                    user_role = user_data[1]
                    user_full_name = user_data[2]

                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.session_state.display_name = user_full_name
                    st.session_state.user_role = user_role
                    
                    st.session_state.agente_usuario_externo = create_agent_for_role(st.session_state.user_role)
                    st.session_state.agente_documentacion = create_documentacion_agent()
                    st.session_state.agent = build_agent_graph(
                        st.session_state.agente_usuario_externo, 
                        st.session_state.agente_documentacion)
                    st.session_state.login_attempt_successful = True
                        
                    st.rerun()
                    return
                else:
                    st.error("Usuario o contraseña incorrectos.")
            except Exception as e:
                st.error(f"Error durante la autenticación: {e}")

def chat_view():
    """Muestra la interfaz de chat con el agente."""
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        # Limpieza de la sesión
        if 'messages' in st.session_state:
            del st.session_state.messages
        if 'agent' in st.session_state:
            del st.session_state.agent
        if 'user_role' in st.session_state:
            del st.session_state.user_role
            
        st.success("Finalizando sesión...")
        time.sleep(1.5)
        st.rerun()

    st.sidebar.markdown(st.session_state.rol_info)

    st.title(f"🩺 Agente de Soporte de Reembolsos")
    
    # Iniciar historial de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Mensaje de bienvenida del agente
        st.session_state.messages.append({"role": "assistant", "content": "¡Hola! Soy tu agente de reembolsos. ¿En qué puedo ayudarte hoy?"})

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de usuario
    if user_input := st.chat_input("Pregunta al agente"):
        # Mostrar mensaje del usuario
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Invocar al agente
        with st.spinner("Pensando en la respuesta..."):
            
            # La configuración de la sesión
            config = {"configurable": {"thread_id": "session_prueba"}} 

            # Prepara el historial de mensajes para el agente
            langchain_messages = [HumanMessage(content=user_input)]
            
            try:
                respuesta = st.session_state.agent.invoke(
                    {"messages": langchain_messages},
                    config=config
                )
                
                # Respuesta del agente
                output = respuesta["messages"][-1].content
                
            except Exception as e:
                output = f"Ocurrió un error al ejecutar el agente: {e}"
                
        # Mostrar respuesta del agente
        with st.chat_message("assistant"):
            st.markdown(output)
        st.session_state.messages.append({"role": "assistant", "content": output})

def logout():
    """Maneja el cierre de sesión."""
    st.session_state.logged_in = False
    del st.session_state.messages
    del st.session_state.agent
    st.success("Sesión finalizada con éxito.")
    time.sleep(1.5)
    st.rerun()

# --- LÓGICA PRINCIPAL ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "loggin_attempt_successful" not in st.session_state:
    st.session_state.login_attempt_successful = False
if st.session_state.logged_in:
    chat_view()
else:
    login_view()