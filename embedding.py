import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_elasticsearch import ElasticsearchStore

# Credenciales de OpenAI
with open("openai.txt") as archivo:
    os.environ["OPENAI_API_KEY"] = archivo.read().strip()

# Credenciales de Elasticsearch
with open("elasticstore.txt") as archivo:
    key_elastic = archivo.read().strip()

# Carga y chunking
loader = PyPDFLoader(file_path="procedimiento_reembolsos.pdf") # Ruta del archivo a indexar
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

separadores = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    " "
]

pdf = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=separadores, is_separator_regex=False)
all_splits = text_splitter.split_documents(pdf)

# Carga a Elasticsearch
vector_store = ElasticsearchStore.from_documents(
    all_splits,
    embedding,
    es_url="http://XX.XX.XX.XX:9200", # Ingresar IP pública del vector store
    es_user="elastic",
    es_password=key_elastic,
    index_name="AAAAAAA") # Ingresar nombre del index
vector_store.client.indices.refresh(index="AAAAAAA") # Ingresar nombre del index

print("Indexación finalizada.")