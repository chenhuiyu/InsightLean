from langchain.document_loaders import UnstructuredPDFLoader,OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


loader=UnstructuredPDFLoader('1-s2.0-S0920410521012511-main.pdf')
data=loader.load()
print(f'You have {len(data)} documents in you data')
print(f'There are {len(data[0].page_content)} characters in your document')


# chunk data up into smaller documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
texts=text_splitter.split_documents(data)
print(f'After split now you have {len(texts)} documents')

# create embeddings of your documents to get ready for semantic search
from langchain.vectorstores import Chroma,Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

embeddings= OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY,environment=PINECONE_ENVIRONMENT)
index_name="insightlearn"
docsearch=Pinecone.from_texts([t.page_content for t in texts],embeddings,index_name=index_name)

query="Who is the author of this paper?"
docs=docsearch.similarity_search(query)

len(docs)
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

llm=OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)
chain=load_qa_chain(llm,chain_type="stuff")
query="Who is the author of this paper?"
docs=docsearch.similarity_search(query,include_metadata=True)
chain.run(input_documents=docs,question=query)
