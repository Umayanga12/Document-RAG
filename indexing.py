import os
import re
import uuid

from safetensors import torch
from transformers import AutoTokenizer, AutoModel

def DocChunking(directory_path:str,model_name:str, para_seperator="/n/n",separator=" "):
    tokenizer = AutoTokenizer.from_pretrainedI(model_name)
    chunk_size = 200
    documents = {}
    all_chunks = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        print(f"file name : {filename}")
        base = os.path.basename(file_path)
        sku = os.path.splitext(base)[0]
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                txt = file.read()

            doc_id = str(uuid.uuid4())
            paragraphs = re.split(para_seperator,txt)

            for paragraph in paragraphs:
                words = paragraph.split(separator)
                current_chunk_str = ""
                chunk = []
                for word in words:
                    if current_chunk_str:
                        new_chunk = current_chunk_str  + separator + word
                    else:
                        new_chunk = current_chunk_str + word
                    if len(tokenizer.tokenize(new_chunk)) <= chunk_size:
                        current_chunk_str = new_chunk
                    else:
                        if current_chunk_str:
                            chunk.append(current_chunk_str)
                        current_chunk_str = word

                if current_chunk_str:
                    chunk.append(current_chunk_str)

                for chunk in chunk:
                    chunk_id = str(uuid.uuid4())
                    all_chunks[chunk_id] = {"text": chunk, "metadata": {"file_name": sku}}
                documents[doc_id] = all_chunks
                return documents


model_name  = "BAAI/bge-small-en-v1.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def map_document_embeddings(documents, tokenizer, model):
    mapped_document_db = {}
    for id, dict_content in documents.items():
        mapped_embeddings = {}
        for content_id, text_content in dict_content.items():
            text = text_content.get("text")
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            mapped_embeddings[content_id] = embeddings
        mapped_document_db[id] = mapped_embeddings
    return mapped_document_db


from qdrant_client.models import PointStruct

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)

# Create collection for embeddings
client.recreate_collection(
    collection_name="document_chunks",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # 384 for bge-small-en
)


def upload_embeddings_to_qdrant(client, embeddings_dict, collection_name="document_chunks"):
    points = []

    for doc_id, chunks in embeddings_dict.items():
        for chunk_id, vector in chunks.items():
            points.append(PointStruct(
                id=chunk_id,
                vector=vector,
                payload={"doc_id": doc_id}
            ))

    client.upsert(collection_name=collection_name, points=points)


def search(query: str, tokenizer, model, client, top_k=5):
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        vector = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()

    hits = client.search(
        collection_name="document_chunks",
        query_vector=vector,
        limit=top_k
    )

    for hit in hits:
        print(f"Score: {hit.score:.3f}, ID: {hit.id}, Payload: {hit.payload}")
