import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import pickle
import PyPDF2
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader, _reader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


#sidebar contents

with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('Welcome to the LLM Chat App! This app is designed to help you chat with the LLM team. Please enter your message in the text box below and click the "Send" button to chat with us. We will respond to your message as soon as possible. Thank you for using the LLM Chat App!')
    st.markdown('---')
    st.markdown('Please enter your message below:')
    message = st.text_area('')
    st.markdown('---')
    st.markdown('Click the "Send" button to chat with us:')
    if st.button('Send'):
        st.write('You: ', message)
        st.write('LLM Team: ', 'Hello! How can we help you today?')

    add_vertcal_space = st.sidebar.empty()
    st.write(' Made with Love')
#main contents

def main():
    st.header('Chat with PDF')
    # st.write('Welcome to the LLM Chat App!')
    load_dotenv()
    #upload document
    pdf = st.file_uploader('Upload a PDF document', type=['pdf'])
    # st.write(pdf.name)
    
    if pdf is not None:
        pdf_reader = PyPDF2.PdfReader(pdf)
        st.write(pdf_reader)

        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)


        #embeddings
        embeddings = OpenAIEmbeddings()

        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
                st.write('Embeddings loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        #query
        query = st.text_input('Enter your query here:')
        st.write('You:', query)
if __name__ == '__main__':
    main()