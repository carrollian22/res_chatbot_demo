from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def build_qa_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
