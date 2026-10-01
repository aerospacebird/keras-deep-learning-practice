#LCEL = Langchain Expression Language
# chain = prompt/model/output_parser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv
template = """
당신은 영어를 가르치는 10년차 영어 선생님 입니다.   # 페르소나
주어진 상황에 맞는 영어회화에  맞는 영어회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

#상황:
{question}

#FORMAT:
-영어회화:
-한글번역:

"""
load_dotenv

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

prompt = PromptTemplate.from_template(template=template)




model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key = api_key,
    base_url= base_url  # "https://monogpt.kr/api/monorouter/v1/"

)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt|model|output_parser 

input= {"question":  "저는 부산 해운대에서 수영을 하고 싶어?"}

response=chain.invoke(input)  # invoke는 model.predict와 동일한 역할을 한다.

print(response)
