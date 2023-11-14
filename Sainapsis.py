from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
import pinecone
import os

@tool("SayHello", return_direct=True)
def say_hello(name: str) -> str:
    """Answer when someone says hello"""
    return f"Hello {name}! My name is Sainapsis"

def load_documents_to_pinecone(docs):
    """Load documents to Pinecone using OpenAI embeddings"""
    dataList = []
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    embeddings = OpenAIEmbeddings()
    for doc in docs:
        with open(doc, 'r', encoding='UTF-8', errors='ignore') as file:
            text = file.read()
        data = Pinecone.from_texts(texts=[text], embedding=embeddings, index_name="documentos")
        dataList.append(data)
    print(dataList)

def search_documents(query):
    """Search for documents in Pinecone based on a query"""
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
    embeddings = OpenAIEmbeddings()
    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index("documentos", embeddings)
    docs = docsearch.similarity_search(query)
    print(docs)

def main():
    llm = ChatOpenAI(temperature=0)
    tools = [say_hello]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    print(agent.run("Hello! My name is Julia"))

if __name__ == "__main__":
    documents = [
        "documents\economia.txt",
        "documents\ingenieria-civil.txt",
        "documents\ingenieria-electrica.txt",
        "documents\ingenieria-electronica.txt",
        "documents\ingenieria-industrial.txt",
        "documents\ingenieria-sistemas.txt"
    ]

    query_to_search = "Cuantos años de acreditación tiene ingeniería de industrial?"

    load_documents_to_pinecone(documents)
    search_documents(query_to_search)
    main()
