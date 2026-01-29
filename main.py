from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_classic.globals import set_debug
import os

set_debug(True)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class Destino(BaseModel):
    cidade: str = Field("Nome da cidade sugerida para visitar")
    motivo: str = Field("Motivo pelo qual a cidade é recomendada")

class Restaurantes(BaseModel):
    cidade: str = Field("Nome da cidade sugerida para visitar")
    restaurantes: str = Field("Restaurantes recomendados na cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()},    
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes em {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()},    
)

modelo = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    temperature=0.5, 
    google_api_key=api_key
)

prompt_cultural = PromptTemplate(
    template="""
   Sugira algumas atrações culturais para visitar em {cidade}.
    """,
)

cadeia1 = prompt_cidade | modelo | parseador_destino
cadeia2 = prompt_restaurantes | modelo | parseador_restaurantes
cadeia3 = prompt_cultural | modelo | StrOutputParser()

cadeia = (cadeia1 | cadeia2 | cadeia3)

resposta = cadeia.invoke(
    {
        "interesse": "praias"
    }
)

print(resposta)