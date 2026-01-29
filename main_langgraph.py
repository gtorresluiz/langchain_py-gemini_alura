from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",    
    temperature=0.5,
    google_api_key=api_key
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um especialista de viagens com destinos para praias brasileiras. Apresente-se como Mestre das Praias."),
        ("human", "{query}")
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um especialista de viagens com destinos para montanhas brasileiras. Apresente-se como Mestre das Montanhas."),
        ("human", "{query}")
    ]
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        "system", "Você é um roteador de consultas de viagens. Direcione a consulta para o especialista adequado com base no destino mencionado (praia ou montanha). Responda apenas com 'praia' ou 'montanha'.",
        ("human", "{query}")
    ]
)

roteador = prompt_roteador | modelo.with_structured_output(Rota)

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"query": estado["query"]}, config)}

async def no_consultor_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}

async def no_consultor_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}

def escolher_no(estado: Estado)->Literal["praia", "montanha"]:
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_consultor_praia)
grafo.add_node("montanha", no_consultor_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {
            "query": "Quero escalar montanhas radicais no sul do Brasil"
        }
    )
    print(resposta["resposta"])

asyncio.run(main())