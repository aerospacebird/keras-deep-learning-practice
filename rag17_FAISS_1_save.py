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
################################### 여기부터 FAISSS ############################################################
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
#faiss_index = faiss.IndexFlatL2(len(1536)
print('FAISS 인덱스 초기화 준비 완료')

#FAISS 백터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d)

faiss_db = FAISS(
    embedding_function=embeddings,
    index = faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},

)
# 저장된 문서의 갯수를 확인.
print(faiss_db.index.ntotal) # 0
################################################# faiss 준비 완료 ####################################################
db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,    
)
DB_PATH = './_db/Faiss17'

db.save_local(
    folder_path=DB_PATH,
    index_name="faiss_index17",
)


# exit()







# DB_PATH = './_db/Chroma/'

# #저장
# db = Chroma.from_documents(
#      documents=split_doc1 + split_doc2,
#      embedding=embeddings,
#      persist_directory=DB_PATH,
#      collection_name='croma11'
# )

# print("Chroma 문서저장 끝")
