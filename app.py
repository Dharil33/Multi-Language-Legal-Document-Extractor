from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import base64
from src.prompt import Prompt_Template

load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)


st.set_page_config(layout='wide',page_title="Legal Information Extractor")

def prompt_template(prompt):
    prompt = PromptTemplate(
    template=prompt, input_variables=["context", "question"]
)
    return prompt
    
def preprocessing(pdf_file):
    page_content = PyPDFLoader(pdf_file)
    pages = page_content.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0
    )
    context = "\n\n".join(str(p) for p in pages)
    texts = text_splitter.split_text(context)
    return texts


def get_qa_chain(filepath,input,stuff_chain):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = preprocessing(filepath)
    vector_index = Chroma.from_texts(texts,embeddings).as_retriever()
    docs = vector_index.get_relevant_documents(input)

    stuff_answer = stuff_chain(
    {
        "input_documents": docs,
        "question": input
    },
    return_only_outputs=True
    )
    return stuff_answer['output_text']

def displaytext(file):
    with open(file,'rb') as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_html = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_html,unsafe_allow_html=True)  


prompt = prompt_template(Prompt_Template)

stuff_chain = load_qa_chain(
    model,
    chain_type="stuff",
    prompt=prompt
    )

def main():
    st.title("Legal Information Extractor")
    upload_file = st.file_uploader("Choose a Pdf: ",type=['pdf'])
    input = st.text_input("Input Prompt: ",key="input")
    if upload_file is not None:
        if st.button("Give Answer:"):
            col1,col2 = st.columns(2)
            filepath = upload_file.name
            with open(filepath,'wb') as temp_file:
                temp_file.write(upload_file.read())
            with col1:
                st.info("Uploaded Legal Document")
                text_viewer = displaytext(filepath)
            with col2:
                st.info("The Response is:")
                ans = get_qa_chain(filepath,input,stuff_chain)
                st.success(ans)

if __name__ == '__main__':
    main()

