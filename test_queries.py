from langchain_openai import ChatOpenAI
import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

# Credenciales OpenAI y PostgreSQL
with open("openai.txt") as archivo:
    os.environ["OPENAI_API_KEY"] = archivo.read().strip()
    
with open("postgresql.txt") as archivo:
    uribd = archivo.read()

# Creamos la conexion a la base de datos
db_data = SQLDatabase.from_uri(uribd)

# Herramienta BD
toolkit_bd = SQLDatabaseToolkit(db=db_data,llm=ChatOpenAI(temperature=0))
tools_bd = toolkit_bd.get_tools()

# Test 1 -> Obtención de nombre completo para display_name
usuario = "USUARIO"
contrasena = "CONTRASEÑA"

sql_query = f"""
    SELECT 
        "usuario",
        "tipousuario", 
        CONCAT("nombres", ' ', "apellidopaterno", ' ', "apellidomaterno") AS NombreCompleto
    FROM usuarios_sistema 
    WHERE 
        "usuario" = '{usuario}' AND 
        "contrasena" = '{contrasena}' AND
        "estado" = 'Activo';
"""
# Test 2 -> Revisión de tabla actualizada (registro o actualizar)
sql_query2 = """
    SELECT * FROM reembolsos
"""

# Ejecutamos query
try:
    resultados = db_data.run(sql_query2)
    print("--- Consulta SQL ejecutada: ---")
    print(sql_query2)
    print("--- Registros de la tabla: ---")
    print(resultados)
    
except Exception as e:
    print(f"Error: {e}")