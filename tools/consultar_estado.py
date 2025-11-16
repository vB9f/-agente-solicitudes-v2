from langchain.tools import tool

# --- Lógica Interna --- #
def consultar_estado_logica(db_conn, n_solicitud: str, usuario: str = None) -> str:
    """
    Consulta el estado de una solicitud de reembolso por número en la base de datos SQL.
    Aplica filtro por usuario_id (ID de login) si se proporciona.
    """
    
    n_solicitud = n_solicitud.strip()
    usuario = usuario.strip() if usuario else None

    # 1. Construir la consulta SQL base
    query = f"""
    SELECT
        "n_solicitud",
        "nomusuario",
        "nombeneficiario",
        "tipogasto",
        "monto",
        "fecharegistro",
        "estado",
        "respuestaequipo"
    FROM reembolsos
    WHERE "n_solicitud" = '{n_solicitud}'
    """
    
    # 2. Agregar el filtro de seguridad por usuario logueado
    if usuario:
        query += f"""
        AND "usuario" = '{usuario}'
        """
        
    query += ";" # Cerrar la consulta

    # 3. Ejecutar la consulta
    try:
        resultados = db_conn.run(query)

        # 4. Verificar si se obtuvieron resultados
        if "No rows" in resultados or not resultados.strip().split('\n')[-1].strip():
            if usuario:
                return f"No se encontró ninguna solicitud con el número **{n_solicitud}** asociada al usuario **{usuario}**."
            else:
                return f"No se encontró ninguna solicitud con el número: **{n_solicitud}**."

        # 5. Formatear la respuesta
        return (
            f"Resultados de la consulta para la Solicitud **{n_solicitud}**:\n"
            f"A continuación se muestra el resultado en formato de tabla (Columna: Valor):\n"
            f"--- RESULTADO DE LA BD ---\n"
            f"{resultados}"
        )
        
    except Exception as e:
        return f"Error al consultar la base de datos SQL. Detalle: {e}"


# --- Función Wrapper para inyección de DB_CONN --- #
def create_tool_consultar_estado(db_conn_instance):
    """
    Crea la herramienta de LangChain, inyectando la instancia activa de la conexión a la BD.
    """
    @tool("consultar_estado")
    def consultar_estado_tool(n_solicitud: str, usuario: str = None) -> str:
        """
        Consulta el estado y detalles de una solicitud de reembolso médico por su número. 
        
        Si se proporciona el 'usuario' (el ID de login del usuario), se verifica que la solicitud 
        pertenezca a ese ID para asegurar el acceso (modo Usuario). 
        Si el 'usuario' es nulo, se permite el acceso total (modo Admin).
        
        Argumentos:
        1. n_solicitud (str): El número de solicitud (Ej: MED_00001, CON_01234).
        2. usuario (str, opcional): El ID de login del usuario que realiza la consulta.
        """
        
        return consultar_estado_logica(
            db_conn_instance,
            n_solicitud,
            usuario
        )
        
    return consultar_estado_tool