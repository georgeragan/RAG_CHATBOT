"""PDF Document Query System Using FAISS and Google Generative AI"""
#Importing Necessary libraries
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.chains import RetrievalQA
import hashlib

#loading env variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key: genai.configure(api_key=api_key)

#creates an instance of the FastAPI class
app = FastAPI()

vector_store = None # global store FAISS

# read text frm pdf. if any error in file handle it
def extract_text_from_pdf(pdf_files: List[UploadFile]) -> str:
 extracted_text = ""
 for pdf_file in pdf_files:
  try:
   pdf_reader = PdfReader(pdf_file.file)
   for page in pdf_reader.pages:
    extracted_text += page.extract_text() or ""  
  except Exception as e: print(f"Error reading PDF file {pdf_file.filename}: {e}")
 return extracted_text

#Divides the extracted text into smaller chunks
def split_text_into_chunks(text: str):
 splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
 return splitter.split_text(text)

#Generates a unique hash for a text chunk to avoid duplicate entries in the FAISS index
def get_chunk_hash(chunk: str) -> str: return hashlib.md5(chunk.encode("utf-8")).hexdigest()

# Builds a FAISS index using the text chunks and associates metadata
def create_faiss_index(chunks: List[str], pdf_files: List[UploadFile]):
 embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
 metadata = []
 unique_chunks = []
 for pdf_file in pdf_files:
  chunk_hashes = set()
  for chunk_index, chunk in enumerate(chunks):
   chunk_hash = get_chunk_hash(chunk)
   if chunk_hash not in chunk_hashes:
    chunk_hashes.add(chunk_hash)
    unique_chunks.append(chunk)
    metadata.append({"source": pdf_file.filename,"chunk_index": chunk_index})
 vector_store = FAISS.from_texts(unique_chunks, embedding=embeddings, metadatas=metadata)
 vector_store.save_local("faiss_index")
 return vector_store

#Sets up a RetrievalQA chain that uses a language model and a retriever to answer user queries
def create_chain(retriever):
 llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
 prompt_template = """Use the following context to answer the question. Provide accurate and concise responses.
 Context: {context}
 Question: {question}
 Answer: """
 prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
 return RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type_kwargs={"prompt": prompt})

#Handles user queries, retrieves relevant documents, and generates an answer with attributed sources
def process_query(user_query: str):
 embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
 try:
  local_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
  retriever = local_store.as_retriever()
  relevant_docs = retriever.invoke(user_query, top_k=7)
  chain = create_chain(retriever)
  response = chain.invoke({"query": user_query})
  answer_text = response.get("result", "").lower()
  ranked_sources = [(doc.metadata.get("source"), doc.metadata.get("chunk_index"), doc.page_content) for doc in relevant_docs]
  unique_sources = []
  added_files = set()
  for source, chunk_index, content in ranked_sources:
   if source and source not in added_files:
    added_files.add(source)
    unique_sources.append(source)
  return response, unique_sources
 except Exception as e:
  return {"result": "Error processing query."}, []

#Handles file uploads, extracts text from PDFs, splits text into chunks, and builds a FAISS index."""
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
 global vector_store
 if files:
  try:
   extracted_text = extract_text_from_pdf(files)
   chunks = split_text_into_chunks(extracted_text)
   vector_store = create_faiss_index(chunks, files)
  except Exception as e: return {"message": f"Error during file upload: {e}"}
 return {"message": "Documents processed successfully." if files else "No files uploaded."}

#Processes user queries, retrieves relevant data, and generates an answer using the FAISS index
@app.post("/chat")
async def chat(question: str = Form(...)):
 global vector_store
 if vector_store:
  try:
   answer, sources = process_query(question)
   return JSONResponse(content={"answer": answer.get("result", "No answer found."),"sources": sources})
  except Exception as e:
   return JSONResponse(content={"answer": "Error processing query.","sources": []}, status_code=500)
 return {"answer": "No documents available. Upload files first.", "sources": []}

#run the fastapi
if __name__ == "__main__":
 import uvicorn
 uvicorn.run(app, host="0.0.0.0", port=8000)