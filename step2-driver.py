import logging, os
from VectorDbPinecone import VectorDbPinecone
import Utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8', 
                    level=os.getenv('DEBUG_LEVEL', 'INFO'))


vdb: VectorDbPinecone = VectorDbPinecone()
vdb.create_or_get_index("driver", dimension=768)
vdb.load_from_file("driver", "data\\driver-full.txt")


while True:
    query = input("Enter a question: ")
    if not query:
        break
    
    r = vdb.similarity_search("driver", query, k=1)
    
    r0text = r.result.hits[0]['fields']['chunk_text']
    logging.info("RETRIEVED CONTEXT >>> %s" % (r0text))

    query_and_context = Utils.make_prompt(query, r0text)
    logging.info("QUESTION TO ASK >>> %s" % (r0text))
    
    response = Utils.ask_gemini(enriched_ask=query_and_context)
    logging.info("RESPONSE RETURNED <<< %s" % response)
    print(response)


vdb.vectorstore.delete_index(name="driver")
