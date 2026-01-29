from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.7, 
    google_api_key=api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf",
]

documentos = sum(
    [
        PyPDFLoader(arquivo).load() for arquivo in arquivos
    ], []
)

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documentos)

dados_recuperados = FAISS.from_documents(
    pedacos,
    embeddings
).as_retriever(search_kwargs={"k": 2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente especializado em seguros de viagem. Responda usando exclusivamente as informações fornecidas para responder às perguntas dos usuários."),
        ("human", "{query}\n\nContexto:\n{context}\n\nResposta:")
    ]
)

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder(pergunta:str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join([um_trecho.page_content for um_trecho in trechos])
    return cadeia.invoke(
        {   
            "query": pergunta,
            "context": contexto
        }   
    )

print(responder("Como devo proceder caso tenha um item comprado roubado, possuindo um cartão gold?"))