from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import os

load_dotenv()

class AnaliseSentimento(BaseModel):
    comentario: str = Field(description="O comentário do cliente")
    sentimento: str = Field(description="O sentimento do texto (positivo, negativo ou neutro)")
    score: float = Field(description="Score do sentimento entre 0 e 10")
    palavras_chave: List[str] = Field(description="Lista de palavras que indicam o sentimento")
    justificativa_score: str = Field(description="Justificativa para o score")

parser = PydanticOutputParser(pydantic_object=AnaliseSentimento)

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-8b-8192",
    temperature=0.3
)

template = """
    Você é um analista de sentimento na seção de comentários de um e-commerce. 
    Analise o seguinte texto e extraia os dados solicitados:
    Texto: {texto}

    A resposta deve ser no seguinte formato:
    {format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["texto", "format_instructions"],
    template=template,
)

chain = prompt | llm

texto_exemplo = "Odiei o resultado do projeto! O trabalho em equipe foi excelente."

resultado_json = chain.invoke(input={"texto": texto_exemplo, "format_instructions": parser.get_format_instructions()})

resultado = parser.parse(resultado_json.content)


print()
print(f"Comentário: {resultado.comentario}")
print(f"Sentimento: {resultado.sentimento}")
print(f"Palavras-chave: {resultado.palavras_chave}")
print(f"Score: {resultado.score}")
print(f"Justificativa: {resultado.justificativa_score}")
print()
