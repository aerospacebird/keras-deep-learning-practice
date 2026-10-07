# 14-2 copy

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"



# 3. 임베딩



from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
        #dimensions=5  # [0.6533203125, -0.389404296875, 0.01232147216796875, 0.31787109375, 0.5654296875] 5개만 가져온다.
)

# sample_text = " 삼성전자의 창업자는 누구인가요?"
# vector = embeddings.embed_query(sample_text)
# print(vector)
# print(len(vector))    #1536
#4. exit()
DB_PATH = './_db/Chroma12/'
#저장
# vector_store = Chroma.from_documents(
#      documents=texts,
#      embedding=embeddings,
#      persist_directory=DB_PATH,
#      collection_name='croma12',
# )

vector_store = Chroma(
     #documents=texts,
     embedding_function=embeddings,
     persist_directory=DB_PATH,
     collection_name='croma12',
)

print(f"벡터 저장소에 저장된 문서 수:{vector_store._collection.count()}",) #벡터 저장소에 저장된 문서 수:156

# query = "삼성전자의 창업주는 누구인가요?"
# result = vector_store.similarity_search(query)

# print(f"검색 결과의 길이:{len(result)}") #검색 결과의 길이:4

####################################### Retrievers##################################################
####################################### 검색기  ####################################################
retriever = vector_store.as_retriever(search_kwarge={"k": 2})
print(retriever)

#aaa = retriever.invoke(query)

# print(f"검색된 관련 문서 수: {len(aaa)}") #검색된 관련 문서 수: 4
# print(f"첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}") # 삼성전자 사업 전망

##################################### model 연결하기 ################################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model= "gpt-5.6-terra",
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

# response= model.invoke("삼성전자 사장은 누구야?")  # model.predict 역할을 한다.invoke
# print("model의 답변:", response.content)
print("===============================")

# query_with_context = f"""
#     {aaa[0].page_content}\n\n
#     위 내용에 근거하여 다음 질문에 답변하세요. \n\n{query}
# """

# response = model.invoke(query_with_context)
# print("model의 응답:", response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template("""


컨텍스트 : {context}  #백터 db에서 데이터를 가져온다.
질문: {input}
답변:

""") 

#체인 만들기

docu_chain = create_stuff_documents_chain(model, prompt) # prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain) #검색 |docu_chain

# # 체인 실행
# query = "삼성전자의 창업자는 누구인가요?"
# response = rag_chain.invoke({"input": query})

# print(response)
# print("=============== KEYS() ============================")
# print(response.keys())
# #dict_keys(['input', 'context', 'answer'])
# print("============== CONTENT =============================")
# print(['context'][0].page_content)

# print("============== ANSWER =============================")
# print(response["answer"])


# exit()
#################################################### Gradio DASHBOARD연결 ###################################################
###############################################################################################################
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({"input": message})
    return response['answer']

#Gradio 인터페이스 만들기

demo = gr.ChatInterface(fn=answer_invoke, title='동욱챗봇!')

# gradio 실행
demo.launch(share=True)  # 외부 URL을 제공하여 준다.
#demo.launch()




#xit()







# #exit()
# split_doc1 = loader1.load_and_split(text_splitter)
# split_doc2 = loader2.load_and_split(text_splitter)

# # 문서 갯수 확인

# #print(split_doc1)
# print(len(split_doc1), len(split_doc2)) # 청크 9 9

# # 문서 embedding
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(
#         model = "text-embedding-3-small",
#         api_key=api_key,
#         base_url=base_url,
#         dimensions=5  # [0.6533203125, -0.389404296875, 0.01232147216796875, 0.31787109375, 0.5654296875] 5개만 가져온다.
# )

# DB_PATH = './_db/Chroma/'

# #저장
# db = Chroma.from_documents(
#      documents=split_doc1 + split_doc2,
#      embedding=embeddings,
#      persist_directory=DB_PATH,
#      collection_name='croma11'
# )

# print("Chroma 문서저장 끝")
