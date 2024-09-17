import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader

load_dotenv()

loader = CSVLoader(file_path="respostas.csv", encoding="ISO-8859-1") 
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    return [doc.page_content for doc in similar_response]

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

template = """

Você é um assistente virtual de um escritório de contabilidade focado em abertura e regularização de igrejas.
Sua função será dar respostas ao meus clientes.
Vou te apresentar um conjunto de perguntas e respostas rotineiras sobre o nosso negócio ou sobre os serviços que prestamos.

Siga todas as regras abaixo:
1/ Você deve buscar se comportar de maneira profissional, focado nas respostas.

2/ Suas respostas devem ser bem similares ou até identicas as encontradas na base de dados, utilizando o tom de voz, argumentos lógicos e demais detalhes.

3/ Foque em apresentar respostas úteis para meus clientes.

Aqui estão perguntas comuns feitas por nossos cliente:
{message}

Aqui está uma lista de respostas rotineiras, sobre o nosso negócio ou sobre os serviços que prestamos. Este histórico de conversa servirá como base para que você compreenda nossos produtos, serviços e modo de atendimento.
{best_practice}

Escreva a melhor resposta que eu deveria enviar para meu potencial cliente.
"""

prompt = PromptTemplate(
    input_variables=["message","best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    try:
        best_practice = retrieve_info(message)
        response = chain.invoke({"message": message, "best_practice": best_practice})
        # Se a resposta vier como um dicionário, extraímos o valor da chave 'text'
        if isinstance(response, dict) and 'text' in response:
            return response['text']
        return response
    except Exception as e:
        return f"Desculpe, houve um erro ao processar sua solicitação: {str(e)}"



def main():
    st.set_page_config(
        page_title="Assistente Comercial"
    )
    st.header("Étika Soluções - Assistente Comercial")
    message = st.text_area("Pergunta:")

    if message:
        st.write("Gerando respostas...")
        result = generate_response(message)
        st.info(result)

if __name__ == '__main__':
    main()