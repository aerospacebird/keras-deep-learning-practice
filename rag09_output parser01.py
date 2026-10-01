#LCEL = Langchain Expression Language
# chain = prompt/model/output_parser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

prompt = PromptTemplate.from_template("{topic}에 대하여 쉽게 설명해 주세요.")




model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key = api_key,
    base_url= base_url  # "https://monogpt.kr/api/monorouter/v1/"

)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt|model|output_parser 

input= {"topic": "그래핀의 원자구성에 대하여 설명해줘?"}

response=chain.invoke(input)  # invoke는 model.predict와 동일하다.

print(response)
