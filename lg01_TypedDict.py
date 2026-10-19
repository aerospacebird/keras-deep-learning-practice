

from typing import Dict, TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AIMessage, HumanMessage



sample_dict = Dict[str, str] = {
    'name': '김말동',
    'age' : '39',
    'job' : '회사원',
}

print(sample_dict)
print(sample_dict['name'])

#####################################################################################

class Person(TypedDict):
    name : str
    age : int
    job : str

person = Person(name='홍길동', age=40, job='도적')
print(person)
###################################################################################

person2: Person ={'name': '김개똥', 'age': 20, 'job':'전사'}









# class MyDate(TypedDict):    #  TypedDict 상속 받겠다는 의미이다.
#     messages : Annotated[list, add_messages]  # 자동 append해준다. 덮어쓰는것이 아니고!!!  #add_mssages가 Reducer

# from langchain_core.messages import AIMessage, HumanMessage

# msg1 = [HumanMessage(content='안녕', id='1')]
# msg2 = [HumanMessage(content='방가', id='2')]

# results = add_messages(msg1, msg2)
# print(results)

# ###################################################################################

# msg3 = [HumanMessage(content='안녕', id='3')]
# msg4 = [HumanMessage(content='안녕', id='4')]

# results2 = add_messages(msg3, msg4)
# print(results2)

