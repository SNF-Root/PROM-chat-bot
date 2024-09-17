from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
import faiss

vector_matrix = np.load('vector_matrix.npy' , allow_pickle= True)

print (vector_matrix.shape)

# Create a FAISS index with the right dimensionality
d = vector_matrix.shape[1]  # vector dimension
index = faiss.IndexFlatL2(d)  # L2 distance for similarity

# Add the vectors to the FAISS index
index.add(vector_matrix)

# Step 3: Integrate with LangChain
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS(index, embeddings)

from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.1:8b",
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

question = "What is peng wei email"

docs = vector_store.similarity_search(question)

print(chain.invoke(docs))