#LCEL = Langchain Expression Language
# chain = prompt/model/output_parser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv

api_key = os.environ["MONOROUTER_API_KEY"].strip()  # 문자열의 양쪽 끝에 있는 불필요한 문자(기본적으로 공백과 줄바꿈)를 제거하는 문자열 메서드입니다.
base_url = "https://monogpt.kr/api/monorouter/v1/"

prompt = PromptTemplate.from_template("{topic}에 대하여 쉽게 {how} 설명해 주세요.")




model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key = api_key,
    base_url= base_url  # "https://monogpt.kr/api/monorouter/v1/"

)

chain = prompt|model  

input= {"topic": "양자 컴퓨팅에 대하여 설명해줘?", "how" : "초등학생도 이해하기 쉽게"}

response=chain.invoke(input)  # invoke는 model.predict와 동일하다.
print(response.content)
