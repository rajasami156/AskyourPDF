import os
import streamlit as st
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import pickle
from PyPDF2 import PdfReader
from langchain.chat_models import ChatOpenAI
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain

api_key = "ENTER YOUR API KEY HERE"
os.environ["OPENAI_API_KEY"] = api_key

with st.sidebar:
    st.title("✨ SwiftQ PDF SAGE")
    st.markdown('''
                ## About
                👉 This  app  is  powered  by  LLM  chatbot 
                Made with ❤ by [SAMIULLAH](https://www.linkedin.com/in/samiullah156/)    (YOU CAN CHANGE WITH YOUR OWN IF YOU WISH TO) 
                ''')
def main():
    st.header("Ask your PDF 📈")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF ", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = ChatOpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)

if __name__ == '__main__':
    main()
