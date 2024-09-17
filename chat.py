from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

question = "Alex Birthday"

loaded_vector_store = Chroma(
    persist_directory="chroma-vectors",  # The directory where Chroma is saved
    embedding_function=local_embeddings
)

print("Chroma vector store loaded successfully.")

docs = loaded_vector_store.similarity_search(question, k=1)
print(len(docs))

from langchain_ollama import ChatOllama

model = ChatOllama(
    model="gemma2:2b",
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "Summarize the main themes in these retrieved docs: {docs}"
)

# Convert loaded documents into strings by concatenating their content
# and ignoring metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = {"docs": format_docs} | prompt | model | StrOutputParser()

#docs = loaded_vector_store.similarity_search(question)

print(chain.invoke(docs))