import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url = "https://monogpt.kr/api/monorouter/v1/"

from glob import glob

#폴더에서 텍스트 파일 목록을 가져오기
path = './_data/rag_data/'

txt_files = glob(os.path.join(path, '*.txt'))

print(txt_files)
#['./_data/rag_data\\2026_AI_for_ALL.txt', './_data/rag_data\\nvidia_outlook.txt'],
# './_data/rag_data\\samsung_outlook.txt']


#01. 데이터 불러온다.
#path = './_data/rag_data/'
data = []

for text_file in txt_files:
    loader = TextLoader(text_file, encoding='utf-8')
    #data.append(loader)
    data += loader.load()

print("========================================")
print(data[0])
print("========================================")

print(len(data))
print(data[0].page_content)

char_count = [len(doc.page_content) for doc in data]
print(char_count) #[8158, 2049, 1898]







# exit()

path = './_data/rag_data/'
loader1 = TextLoader(path + "samsung_outlook.txt", encoding="utf-8")
loader2 = TextLoader(path + "nvidia_outlook.txt", encoding="utf-8")

#02. 데이터 자른다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300, # 문단, 줄바꿈,문장, 공백, 문자 같은 **구분자(separator)**를 우선적으로 사용해서 자연스럽게 나눕니다.
    chunk_overlap=10,
    separators=["\n\n", "\n", "", ""], #통상 디폴트. # 문단바꿈 : "\n\n"
)

texts =text_splitter.split_documents(data)
print("생성된 텍스트 청크수 :", len(texts))
print("각청크의 길이 :", list(len(text.page_content) for text in texts))
#각청크의 길이 : [259, 282, 282, 128, 276, 214, 158, 249, 262, 291, 268, 182, 286, 295, 182, 162, 283, 286, 257, 235, 214,
#  258, 207, 286, 220, 198, 271, 57, 272, 122, 9, 269, 299, 284, 289, 209, 222, 230, 254, 249, 299, 90, 247, 243, 185,
#  219, 239, 235, 298, 282, 187, 249]
# print("첫번째 청크의 내용:", texts[0].page_content)
# print("첫번째 청크의 길이:", len(texts[0].page_content))
# print("두번째 청크의 내용:", texts[1].page_content

# 3. 임베딩

# 첫번째 청크의 내용: page_content='2026년 한국의 AI for All 프로젝트와 생성형 AI 서비스 확산

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
        #dimensions=5  # [0.6533203125, -0.389404296875, 0.01232147216796875, 0.31787109375, 0.5654296875] 5개만 가져온다.
)

sample_text = " 삼성전자의 창업자는 누구인가요?"
vector = embeddings.embed_query(sample_text)
print(vector)
print(len(vector))    #1536
#4. exit()
DB_PATH = './_db/Chroma12/'
#저장
vector_store = Chroma.from_documents(
     documents=texts,
     embedding=embeddings,
     persist_directory=DB_PATH,
     collection_name='croma12',
)

print(f"벡터 저장소에 저장된 문서 수:{vector_store._collection.count()}",) #벡터 저장소에 저장된 문서 수:156

query = "삼성전자의 창업주는 누구인가요?"
result = vector_store.similarity_search(query)

print(f"검색 결과의 길이:{len(result)}") #검색 결과의 길이:4

####################################### Retrievers##################################################
####################################### 검색기  ####################################################
retriever = vector_store.as_retriever(search_kwarge={"k": 2})
print(retriever)
aaa = retriever.invoke(query)
print(f"검색된 관련 문서 수: {len(aaa)}") #검색된 관련 문서 수: 4
print(f"첫번째 관련 문서 내용 미리보기: {aaa[0].page_content[:50]}") # 삼성전자 사업 전망

exit()



split_doc1 = loader1.load_and_split(text_splitter)
split_doc2 = loader2.load_and_split(text_splitter)

# 문서 갯수 확인

#print(split_doc1)
print(len(split_doc1), len(split_doc2)) # 청크 9 9

# 문서 embedding
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
        model = "text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
        dimensions=5  # [0.6533203125, -0.389404296875, 0.01232147216796875, 0.31787109375, 0.5654296875] 5개만 가져온다.
)

DB_PATH = './_db/Chroma/'

#저장
db = Chroma.from_documents(
     documents=split_doc1 + split_doc2,
     embedding=embeddings,
     persist_directory=DB_PATH,
     collection_name='croma11'
)

print("Chroma 문서저장 끝")
