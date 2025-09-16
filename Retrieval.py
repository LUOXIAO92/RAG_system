from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

class Retrieval:
    def __init__(
            self,
            vector_db       : str,
            collection_name : str,
            embedding_model : str,
            ):
        
        self.vector_db       = vector_db
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self.client    = None
        self.embedding = None

    def load(self):
        self.client = MilvusClient(self.vector_db)
        self.embedding = SentenceTransformer(self.embedding_model)

    def search_by_query(self, queries : list[str], limit : int = 10):

        queries_encoded = self.embedding.encode(queries)

        results_all_queries = self.client.search(
            collection_name = self.collection_name,
            data            = queries_encoded,
            limit           = limit,
            output_fields   = ["path", "type", "title","text"]
        )

        results_reorganized = []
        for results_one_query in results_all_queries:
            result4query = {}
            for result in results_one_query:
                if result["path" ] not in result4query.keys():
                    result4query[result["path" ]] = {
                        "distance": [result["distance"]],
                        "type": result["type" ], 
                        "title": result["title"], 
                        "text": result["text" ].lstrip(result["title"] + " ")
                        }
                else:
                    result4query[result["path" ]]["text"] += " " + result["text" ].lstrip(result["title"] + " ")
                    result4query[result["path" ]]["distance"].append(result["distance"])
            results_reorganized.append(result4query)


        results_reorganized_list = []
        for results in results_reorganized:
            assert type(results) == dict
            for path in results.keys():
                assert type(results[path]) == dict
                #f"similarity:{sum(results[path]["distance"]) / len(results[path]["distance"])} " + \
                result_str = f"path:{path} " + \
                             f"type:{results[path]["type"]} " + \
                             f"title:{results[path]["title"]} " + \
                             results[path]["text"]
                results_reorganized_list.append(result_str)    

        return results_reorganized_list

