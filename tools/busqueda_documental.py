from langchain.tools import tool

def create_tool_busqueda_documental(vector_store):
    """
    Crea y devuelve la herramienta 'busqueda_documental' vinculada al vector store.
    """
    
    @tool
    def busqueda_documental(pregunta: str) -> str:
        """Busca en la base de datos vectorial de reembolsos para obtener contexto sobre procedimientos, 
        políticas, requisitos, pasos o información general. Útil para responder preguntas teóricas."""
        
        # Usar el Vector Store
        if vector_store is None:
            return "ERROR: La base de datos de documentación no está disponible."
            
        # Búsqueda de similitud
        docs = vector_store.similarity_search_with_score(pregunta, k=3)
        
        contexto = []
        
        for doc, score in docs:
            if score > 0.7: 
                # Se utiliza doc.metadata.get('source', 'N/A') para incluir la fuente del PDF
                contexto.append(f"Contexto: {doc.page_content} (Fuente: {doc.metadata.get('source', 'N/A')})")
                
        if contexto:
            return "\n---\n".join(contexto)
        else:
            return "No se encontró información relevante sobre ese tema en la documentación. Responde al usuario que no tienes ese detalle."

    return busqueda_documental