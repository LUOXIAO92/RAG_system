from Retrieval import Retrieval
from Language_Model import Language_Model, messages_generator

class RAG:
    def __init__(
            self,
            vector_db       : str  ,
            collection_name : str  ,
            embedding_model : str  ,
            model_name      : str  ,
            temperature     : float,
            load_right_now  = False
            ):
        
        self.vector_db       = vector_db      
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.model_name      = model_name     
        self.temperature     = temperature    

        self.retrieval = None
        self.llm       = None
        if load_right_now:
            self.load_models()
            

    def load_models(self):
        if self.llm is None:
            self.llm = Language_Model(
                model_name  = self.model_name  ,
                temperature = self.temperature
            )
            self.llm.load()

        if self.retrieval is None:
            self.retrieval = Retrieval(
                vector_db       = self.vector_db       ,
                collection_name = self.collection_name ,
                embedding_model = self.embedding_model
                )
            self.retrieval.load()

        print("Vector database, embedding model and large lange model have been loaded successfully.")

    def generate(
            self,
            query           : str,
            enable_thinking : bool = False,
            use_rag         : bool = False,
            rag_limit       : int  = 20
            ):
        
        retrieval_results = None
        if use_rag:
            retrieval_results = self.retrieval.search_by_query([query], rag_limit)
        
        messages = messages_generator(query, retrieval_results)

        answer, consumed_tokens = self.llm.generate(messages, enable_thinking)

        return answer, consumed_tokens