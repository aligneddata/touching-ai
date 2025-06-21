import time
import os
from pinecone import Pinecone, ServerlessSpec
from VectorDbIntf import VectorDbIntf
import Utils

class VectorDbPinecone(VectorDbIntf):
    def __init__(self):
        super().__init__()
        self.vectorstore = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    def create_or_get_index(self, index_name: str, dimension: int):
        cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
        region = os.environ.get('PINECONE_REGION') or 'us-east-1'
        spec = ServerlessSpec(cloud=cloud, region=region)

        if index_name not in self.vectorstore.list_indexes().names():
            self.vectorstore.create_index_for_model(
                name=index_name,
                cloud="aws",
                region="us-east-1",
                embed={
                    "model":"llama-text-embed-v2",
                    "field_map":{"text": "chunk_text"}
                }
            )
            while not self.vectorstore.describe_index(index_name).status['ready']:
                time.sleep(1)
        
    def load_from_file(self, index_name: str, filename: str):
        contents = Utils.read_file_to_string(filename)
        chunks = Utils.split_into_chunks(contents, chunk_size=Utils.CHUNK_SIZE)
        index = self.vectorstore.Index(index_name)  
        records = []
        BATCH_SIZE = 50
        for i, d in enumerate(chunks):
            item = {}
            item['_id'] = "rec" + str(i)
            item['chunk_text'] = d
            item['category'] = 'training'
            records.append(item)
            if len(records) == BATCH_SIZE:
                index.upsert_records("ns1", records)
                records = []       
        
    def load_from_texts(self, index_name: str, list_of_texts):
        pass
    
    def similarity_search(self, index_name: str, query: str, k=4):
        results = self.vectorstore.Index(index_name).search(
            namespace="ns1",
            query={
                "top_k": k,
                "inputs": {
                    'text': query
                }
            }
        )
        return results

