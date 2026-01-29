import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.7, 
    google_api_key=api_key
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um guia de viagens especializado em destinos brasileiros. Apresente-se como Mestre dos Passeios."),
        ("placeholder", "{historico}"),
        ("human", "{query}")
    ]
)

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "langchain-py-openai_alura"

def historico_por_sessao(sessao : str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_variable_name="historico"
)

for pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "query": pergunta
        },
        config={"session_id": sessao}
    )
    print(f"Pergunta: {pergunta}\nResposta: {resposta}\n")