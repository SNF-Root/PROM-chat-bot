from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import os


# Set the directory path where your text files are stored
directory_path = '/Users/adenton/Desktop/Local-LLM/docs'

# Get all .txt files from the directory
text_files = glob.glob(os.path.join(directory_path, "*.txt"))

# Initialize a list to hold all documents
documents = []

# Loop through each file and load it
for file_path in text_files:
    loader = TextLoader(file_path)
    # Load the document into Langchain's document format
    doc = loader.load()
    documents.extend(doc)  # Add to the list of documents

# Now `documents` contains all loaded documents
print(f"Loaded {len(documents)} documents.")

# Use a text splitter to break long documents into smaller chunks (if necessary)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Split the documents
split_docs = text_splitter.split_documents(documents)

print(f"Split into {len(split_docs)} document chunks.")

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

persist_directory = "chroma-vectors"
vectorstore = Chroma.from_documents(documents=split_docs, embedding=local_embeddings, persist_directory=persist_directory)

print("chroma index saved.")