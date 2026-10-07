# 11-1 copy하였다.

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
#pip install faiss-cpu
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore #유클리드 거리 유사도

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

# #01. 데이터 불러온다.
# path = './_data/rag_data/'
# loader1 = TextLoader(path + "samsung_outlook.txt", encoding="utf-8")
# loader2 = TextLoader(path + "nvidia_outlook.txt", encoding="utf-8")

# #02. 데이터(문서) 자른다.
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 300,
#     chunk_overlap=100,
#     separators=["\n\n", "\n", "", ""], #통상 디폴트.
# )

# split_doc1 = loader1.load_and_split(text_splitter)
# split_doc2 = loader2.load_and_split(text_splitter)

# # 문서 갯수 확인

# #print(split_doc1)
# print(len(split_doc1), len(split_doc2)) # 청크 9 9

# 문서 embedding
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
        dimensions=5  # [0.6533203125, -0.389404296875, 0.01232147216796875, 0.31787109375, 0.5654296875] 5개만 가져온다.
)
# ################################### 여기부터 FAISSS ############################################################
# faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
# #faiss_index = faiss.IndexFlatL2(len(1536)
# print('FAISS 인덱스 초기화 준비 완료')

# #FAISS 백터 저장소의 벡터 차원 수 (임베딩 차원 수)
# print(faiss_index.d)

# faiss_db = FAISS(
#     embeddings_function=embeddings,
#     index = faiss_index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={},

# )
# # 저장된 문서의 갯수를 확인.
# print(faiss_db.index.ntotal) # 0
################################################# faiss 준비 완료 ####################################################
# db = FAISS.from_documents(
#     documents=split_doc1 + split_doc2,
#     embedding=embeddings,    
# )
DB_PATH = './_db/Faiss17'
# db.save_local(
#     folder_path=DB_PATH,
#     index_name="faiss_index17",
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name="faiss_index17",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

print("========================================================")
#문서 저장소 ID확인
print(db.index_to_docstore_id)
print("========================================================")
# 저장된 결과 확인
print(db.docstore._dict)
print("========================================================")
# 유사도 검색
aaa = db.similarity_search("삼성전자 창업주에 대해 알려줘", k=2,)
print(aaa)
# exit()
####################################### Retrievers##################################################
####################################### 검색기  ####################################################
retriever = db.as_retriever(search_kwarge={"k": 2})
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






# DB_PATH = './_db/Chroma/'

# #저장
# db = Chroma.from_documents(
#      documents=split_doc1 + split_doc2,
#      embedding=embeddings,
#      persist_directory=DB_PATH,
#      collection_name='croma11'
# )

# print("Chroma 문서저장 끝")
