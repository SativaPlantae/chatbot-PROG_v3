import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 🔐 Chave da OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def carregar_qa_chain():
    caminho_pdf = "40.pdf"
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documentos)

    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    retriever = vectorstore.as_retriever()

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Você é um assistente virtual treinado com base em um documento técnico de licenciamento ambiental. Seu estilo é natural, amigável e direto, como se estivesse conversando com alguém em um chat. 

Quando responder, use uma linguagem simples e acessível, como o ChatGPT faria. Seja claro, mas não precisa ser excessivamente formal. Evite repetir demais o conteúdo da pergunta.

Se a resposta não estiver presente no documento, diga algo como: "Hmm, isso não está muito claro por aqui, mas posso tentar ajudar com base no que tenho."

Se a pergunta estiver fora do escopo do documento, diga isso de forma simpática.

-------------------
{context}

Pergunta: {question}
Resposta:"""
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=500,
        openai_api_key=openai_api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    return qa_chain

# 🌐 Interface do app
st.set_page_config(page_title="Chatbot Institucional - Sativa Plantae", page_icon="🤖")
st.title("🤖 Chatbot da AD nº 43/2024")
st.markdown("Converse sobre o conteúdo da Autorização Direta 📄")

# Inicializa o histórico de conversa
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

qa_chain = carregar_qa_chain()

# Formulário de envio
with st.form(key="formulario_chat"):
    user_input = st.text_input("Você:", placeholder="Digite sua pergunta aqui...")
    submit = st.form_submit_button("Enviar")

# Processamento da entrada
if submit and user_input:
    with st.spinner("Consultando o modelo..."):
        try:
            resposta = qa_chain.run(user_input)
            st.session_state.mensagens.append(("Você", user_input))
            st.session_state.mensagens.append(("Chatbot", resposta))
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

# Exibição do histórico
for remetente, mensagem in st.session_state.mensagens:
    if remetente == "Você":
        st.markdown(f"**🧑 {remetente}:** {mensagem}")
    else:
        st.markdown(f"**🤖 {remetente}:** {mensagem}")
