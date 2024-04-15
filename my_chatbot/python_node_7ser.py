
from promptflow.core import tool

import os
import re
import requests
import sys
from num2words import num2words
import os
import pandas as pd
import numpy as np
import tiktoken
from openai import AzureOpenAI


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: str) -> str:
    df=pd.read_csv(os.path.join(os.getcwd(),'sample_data.csv')) # This assumes that you have placed the bill_sum_data.csv in the same directory you are running Jupyter Notebooks
    # return 'hello ' + input1
    df_bills = df[['상품명', '질문내용', '답변내용', '문제유형']]
    df_bills

    pd.options.mode.chained_assignment = None #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#evaluation-order-matters

    # s is input text
    def normalize_text(s, sep_token = " \n "):
        s = re.sub(r'\s+',  ' ', s).strip()
        s = re.sub(r". ,","",s)
        # remove all instances of multiple spaces
        s = s.replace("..",".")
        s = s.replace(". .",".")
        s = s.replace("\n", "")
        s = s.strip()

        return s

    df_bills['질문내용']= df_bills["질문내용"].apply(lambda x : normalize_text(x))

    tokenizer = tiktoken.get_encoding("cl100k_base")
    df_bills['n_tokens'] = df_bills["질문내용"].apply(lambda x: len(tokenizer.encode(x)))
    df_bills = df_bills[df_bills.n_tokens<8192]
    len(df_bills)

    # 샘플 질문 내용
    sample_encode = tokenizer.encode(df_bills.질문내용[0]) 
    decode = tokenizer.decode_tokens_bytes(sample_encode)

    client = AzureOpenAI(
    api_key = 'e841e94785a24b258f283caaccb0c1c1',  
    api_version = "2024-02-01",
    azure_endpoint = 'https://prompton-team-1301.openai.azure.com/'
    )

    def generate_embeddings(text, model="text-embedding-ada-002"): # model = "deployment_name"
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    df_bills['ada_v2'] = df_bills["질문내용"].apply(lambda x : generate_embeddings (x, model = 'text-embedding-ada-002')) # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_embedding(text, model="text-embedding-ada-002"): # model = "deployment_name"
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    def search_docs(df, user_query, top_n=4, to_print=False):
        embedding = get_embedding(
            user_query,
            model="text-embedding-ada-002" # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
        )
        df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

        res = (
            df.sort_values("similarities", ascending=False)
            .head(top_n)
        )
        if to_print:
            print(res)
        return res

    res = search_docs(df_bills, input1, top_n=4)

    
    print(res['질문내용'])
    return res.to_json(orient='records', force_ascii=False)
    # return '' + df
