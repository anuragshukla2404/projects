import os
import streamlit as st
import time
import pickle
from joblib import dump
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.title("AI News Research Tool")
st.sidebar.title("AI Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_urls_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()

llm = OpenAI(temperature=0.9, max_tokens=500)

if process_urls_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text('Data Loading...')
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators= ['\n\n','\n','.',','],
        chunk_size = 1000
    )
    main_placeholder.text('Text Splitter is started...')
    docs = text_splitter.split_documents(data)
    # embeddings and saving data to FAISS index
    embedding = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embedding)
    main_placeholder.text('Embedding Vector Started Building...')
    time.sleep(2)

    vectorstore_openai.save_local("faiss_index")

query = main_placeholder.text_input("Question: ")

file_path = "faiss_index"
embedding = OpenAIEmbeddings()

if query:
    if os.path.exists(file_path):
            vectorstore = FAISS.load_local(
                "faiss_index", embedding,
                allow_dangerous_deserialization=True
            )
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever = vectorstore.as_retriever())
            st.write("Calling LLM...")
            result = chain({"question": query}, return_only_outputs=True)
            # Display Answers
            st.subheader("Answer:")
            st.write(result["answer"])

            # Display sources if available
            sources = result.get("Sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
else:
    st.error("No existing knowledgebase found")

            



    
