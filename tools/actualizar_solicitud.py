import datetime
from langchain.tools import tool

# --- Lógica Interna --- #
def actualizar_solicitud_logica(db_conn, n_solicitud: str, nuevo_estado: str, nueva_respuesta: str) -> str:
    """
    Actualiza el Estado, FechaRespuesta y la RespuestaEquipo de una solicitud de reembolso médica 
    registrada en la base de datos SQL.
    """
    
    # 1. Normalizar la entrada de estado y verificar validez
    nuevo_estado = nuevo_estado.strip().capitalize()
    estados_validos = ["Pendiente", "Aprobado", "Rechazado", "Observado"]
    if nuevo_estado not in estados_validos:
        return f"Estado no válido. Debe ser uno de los siguientes: {', '.join(estados_validos)}"

    fecha_respuesta = datetime.date.today().strftime("%Y-%m-%d")

    # 2. Generar consulta de actualización (UPDATE)
    query = f"""
    UPDATE reembolsos 
    SET 
        estado = '{nuevo_estado}',
        respuestaequipo = '{nueva_respuesta}',
        fecharespuesta = '{fecha_respuesta}'
    WHERE "n_solicitud" = '{n_solicitud}';
    """
    
    # 3. Ejecutar consulta
    try:
        db_conn.run(query)

        query_check = f"""
        SELECT 1 FROM Reembolsos WHERE "n_solicitud" = '{n_solicitud}';
        """
        
        check_result = db_conn.run(query_check)
        
        if "No rows" in check_result or not check_result.strip().split('\n')[-1].strip():
             return f"No se encontró ninguna solicitud con el número: {n_solicitud}."

        return (
            f"Solicitud **{n_solicitud}** actualizada con éxito:\n"
            f"- Nuevo Estado: **{nuevo_estado}**\n"
            f"- Nueva Respuesta: **{nueva_respuesta}**"
        )
    
    except Exception as e:
        return f"Error al actualizar solicitud en la base de datos SQL. Detalle: {e}"


# --- Función Wrapper para inyección de DB_CONN --- #
def create_tool_actualizar_solicitud(db_conn_instance):
    """
    Crea la herramienta de LangChain, inyectando la instancia activa de la conexión a la BD.
    """
    @tool("actualizar_solicitud")
    def actualizar_solicitud_tool(n_solicitud: str, nuevo_estado: str, nueva_respuesta: str) -> str:
        """
        Actualiza el Estado, FechaRespuesta y la RespuestaEquipo de una solicitud de reembolso médica registrada.
        
        El 'nuevo_estado' debe ser uno de: Pendiente, Aprobado, Rechazado, Observado.
        La 'nueva_respuesta' es el comentario/justificación del equipo médico.
        
        Si bien las variables indican 'nuevo' en su denominación, si el usuario solo indica 'estado' o 'respuesta' se debería asociar a ellas.
        """
        
        return actualizar_solicitud_logica(
            db_conn_instance,
            n_solicitud,
            nuevo_estado,
            nueva_respuesta
        )
        
    return actualizar_solicitud_tool