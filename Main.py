from llama_parse import LlamaParse
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import nest_asyncio
from flask import Flask, jsonify,request
from flask_cors import CORS
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
import os

nest_asyncio.apply()
load_dotenv()

app=Flask(__name__)
CORS(app,resources={r"/*":{"origins":"*"}})

#load LLM model
llm=Groq(model="llama3-70b-8192",api_key=os.getenv("GROQ_API_KEY"))
Settings.llm=llm
print("Loading Model Success")

#Model for Embedding vector Conversion
embed_model=FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model=embed_model


def initialize_parser():
    return LlamaParse(
        api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
        parsing_instruction=""" 
        You are parsing the 'Noun Modifier report' dataset. The column 'Noun' represents the primary material name. 

        Please extract the following information from the columns:

        - **Noun**: The primary material name.
        - **Modifier**: The descriptive term modifying the noun.
        - **NM Abbreviation**: The abbreviation for the Noun-Modifier combination.
        - **NM Definition**: The definition of the Noun-Modifier combination.
        - **ImageId**: The reference identifier for the associated image.
        - **Category**: The classification of the material.
        - **Attribute**: The characteristic linked to the noun-modifier.
        - **Attribute Definition**: The explanation of the attribute.
        - **Mandatory**: Specifies whether the attribute is required.
        - **UOM Mandatory**: Indicates if the unit of measurement is required.
        - **Short Sequence**: The short-form sequence identifier.
        - **Long Sequence**: The long-form sequence identifier.
        - **Value**: The assigned value for the attribute.
        - **UOM**: The unit of measurement corresponding to the value.
        """,
        result_type="markdown"
    )

def initialize_qdrant_client():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        prefer_grpc=True,
        api_key=os.getenv("QDRANT_API_KEY")
    )

def parse_excel():
    parser = initialize_parser()
    file_extractor={".xlsx":parser}
    return SimpleDirectoryReader(input_files=['report.xlsx'], file_extractor=file_extractor).load_data() #Will return Document Object

def initialize_vector_store(qdrant_client, collection_name):
    return QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

def save_vector_embeddings_to_qdrant(collection_name):
    qdrant_client=initialize_qdrant_client()
    documents=parse_excel()

    #collection_name=noun_modifier_attributes
    vector_store = initialize_vector_store(qdrant_client,collection_name)
    storage_context=StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)

def create_query_engine(collection_name):
    qdrant_client = initialize_qdrant_client()

    vector_store=initialize_vector_store(qdrant_client,collection_name)
    db_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return db_index.as_query_engine() #returns query_engine

def get_response(query):
 response =create_query_engine("noun_modifier_attributes").query(query)
 return response.response

@app.route("/generate",methods=["POST"])
def generate_response():
    request_body=request.get_json()
    if not request_body:
        return jsonify({"error": "Prompt is required"}), 400

    query=request_body.get("message")
    return jsonify({"response": get_response(query)})

if __name__ == "__main__":
    app.run(debug=True,host="0.0.0.0")