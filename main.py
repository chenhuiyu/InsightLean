import os

import pinecone
import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]


DEFAULT_PDF_FILE = 'https://raw.githubusercontent.com/chenhuiyu/InsightLean/main/1-s2.0-S0920410521012511-main.pdf'

st.title("InsightLean")
uploaded_file = st.file_uploader(
    "Upload a PDF file or use the default file", type=['pdf'])


if uploaded_file:
    loader = UnstructuredPDFLoader(uploaded_file)
else:
    loader = OnlinePDFLoader(DEFAULT_PDF_FILE)


with st.spinner("Loading and processing the document..."):
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    index_name = "insightlearn"
    docsearch = Pinecone.from_texts(
        [t.page_content for t in texts], embeddings, index_name=index_name)

query = st.text_input("Enter your query:",
                      value="Who is the author of this paper?")
# model_option = st.radio("Choose a model:", ("GPT-4", "GPT3.5"))

if query:
    with st.spinner("Searching for the answer..."):
        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

        chain = load_qa_chain(llm, chain_type="stuff")
        docs = docsearch.similarity_search(query, include_metadata=True)
        answer = chain.run(input_documents=docs, question=query)

    st.write(f"Answer: {answer}")
else:
    st.write("Please enter a query.")
