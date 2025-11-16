import datetime
from langchain.tools import tool

# --- Lógica Interna --- #
def registrar_solicitud_logica(db_conn, usuario: str, nombre_asegurado: str, tipo_gasto: str, monto: float, nombre_beneficiario: str = None) -> str:
    
    # 1. Preparación de datos
    tipo_gasto = tipo_gasto.strip().capitalize()
    prefijos = {"Medicinas": "MED", "Exámenes": "EXA", "Consultas": "CON"}
    prefijo = prefijos.get(tipo_gasto, "OTR")
    
    # Reemplazar variables
    usuario = usuario
    usuario_completo = nombre_asegurado
    beneficiario_final = nombre_beneficiario if nombre_beneficiario else usuario_completo
    fecha_registro = datetime.date.today().strftime("%Y-%m-%d")
    
    # 2. Generar código de solicitud (en SQL)
    query_ult_num = f"""
    SELECT COALESCE(MAX(CAST(SUBSTRING("n_solicitud", 5) AS INTEGER)), 0)
    FROM reembolsos
    WHERE "tipogasto" = '{tipo_gasto}';
    """
    
    try:
        ult_num_str = db_conn.run(query_ult_num).strip().split('\n')[-1].strip().split('|')[-1].strip()
        ult_num = int(ult_num_str) if ult_num_str else 0
    except Exception as e:
        ult_num = 0
    nuevo_codigo = f"{prefijo}_{ult_num + 1:05d}"
    
    # 3. Insertar solicitud a tabla en base de datos SQL
    query = f"""
    INSERT INTO reembolsos 
    ("n_solicitud", "usuario", "nomusuario", "nombeneficiario", "tipogasto", "monto", "estado", "fecharegistro", "fecharespuesta", "respuestaequipo")
    VALUES (
        '{nuevo_codigo}', 
        '{usuario}', 
        '{usuario_completo}', 
        '{beneficiario_final}', 
        '{tipo_gasto}', 
        {monto}, 
        'Pendiente', 
        '{fecha_registro}', 
        NULL, 
        'En revisión por el área de Reembolsos'
    );
    """
    
    # 4. Ejecutar consulta
    try:
        db_conn.run(query)
        return f"Solicitud registrada en el sistema con el código: {nuevo_codigo}."
    
    except Exception as e:
        return f"Error al registrar solicitud en el sistema. Verifique que el usuario '{usuario}' exista y que la tabla 'reembolsos' esté creada. Detalle: {e}"


# --- Función Wrapper para inyección de DB_CONN --- #
def create_tool_registrar_solicitud(db_conn_instance):
    """
    Crea la herramienta de LangChain, inyectando la instancia activa de la conexión a la BD.
    """
    @tool("registrar_solicitud")
    def registrar_solicitud_tool(usuario: str, nombre_asegurado: str, tipo_gasto: str, monto: float, nombre_beneficiario: str = None) -> str:
        """
        Registra una nueva solicitud de reembolso médico en la base de datos SQL.'
        
        Argumentos:
        1. usuario (str): Usuario logueado en sistema/plataforma.
        1. nombre_asegurado (str): Nombre completo del titular del beneficio.
        2. tipo_gasto (str): Tipo de gasto (Medicinas, Exámenes, Consultas).
        3. monto (float): Monto del gasto.
        4. nombre_beneficiario (str, opcional): Nombre de la persona que recibió el servicio (si es un dependiente o es diferente al asegurado).

        El nombre del Asegurado se toma automáticamente del contexto.

        El número de solicitud depende del tipo de gasto:
        - Medicinas → prefijo 'MED'
        - Exámenes → prefijo 'EXA'
        - Consultas → prefijo 'CON'
        Si el usuario no te ha compartido alguno de los datos necesarios para registrar la solicitud, volver a preguntar por los datos pendientes.
        """
        
        return registrar_solicitud_logica(
            db_conn_instance,
            usuario,
            nombre_asegurado,
            tipo_gasto,
            monto,
            nombre_beneficiario
        )
    return registrar_solicitud_tool