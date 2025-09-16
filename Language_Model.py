from transformers import AutoModelForCausalLM, AutoTokenizer

class Language_Model:
    def __init__(
            self, 
            model_name      : str,
            temperature     : float | None = 0.4,
            ):
        
        self.model_name  = model_name
        self.temperature = temperature        

        self.tokenizer  = None
        self.model      = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype = "auto",
            device_map  = "auto"
        )

    def generate(
            self, 
            messages        : list[dict], 
            enable_thinking : bool = False
            ):

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize              = False,
            add_generation_prompt = True,
            enable_thinking       = enable_thinking  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        model_outputs = self.model.generate(
            **model_inputs,
            max_new_tokens  = 32768,
            temperature     = self.temperature
            )
        output_ids = model_outputs[0][len(model_inputs.input_ids[0]):].tolist() 

        consumed_tokens = model_outputs[0].size()[0]

        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        output = ""
        assert type(thinking_content) == str
        if len(thinking_content.replace("</think>", "").replace(" ", "")) > 0:
            output += "思考:\n" + thinking_content + "\n"
        output += "回答:\n" + content

        return output, consumed_tokens
    

def messages_generator(query : str, retrieval_results : dict | None):

    if retrieval_results is not None:
        retrieval_results_str = ""
        for key in retrieval_results.keys():
            rag_result = retrieval_results[key]
            retrieval_results_str += f"({key})" + rag_result + ". "

        prompt = "あなたは賢いアシスタントだ。質問に対して、与えられた検索結果に基づいて回答しなさい。" + \
                 "まだ、与えられた検索結果にない内容は勝手に生成しないこと。回答の最後にドキュメントのpathを追加しなさい。" + \
                 f"質問: {query} \n" + \
                 f"検索結果: {retrieval_results_str}"
                 
    else:
        prompt = query

    messages = [
        {
            "role": "user", 
            "content": prompt
        } 
        ,{
            "role": "system", 
            "content": "あなたは賢いアシスタントで、質問を回答しなさい。"
            #"content": "あなたは機密をきちんと守るアシスタントだ。会社の機密情報を話してはいけない。"
        }
        #,{
        #    "role": "assistent", 
        #    "content": "次のような形式で回答しなさい:\n" + \
        #         "質問: \n" + \
        #         "回答: \n" + \
        #         "path: \n"
        #}
    ]

    return messages