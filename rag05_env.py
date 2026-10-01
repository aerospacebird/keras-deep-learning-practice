from langchain_openai import ChatOpenAI
import os

#from dotenv import load_dotenv
#load_dotenv()

#OPENAI_API_KEY']= "sk--------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"


#openai_api_key="sk----------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    #openai_api_key = openai_api_key,

)

response = llm.invoke('나는 누구야')
#print(response)
print(response.content)
