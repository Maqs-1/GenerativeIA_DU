from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def get_rag_response(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Définition du système de réponse
    system_prompt = (
        "Utilise les éléments de contexte suivants pour répondre à la question. "
        "Si tu ne sais pas, dis que tu ne sais pas. "
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Création de la chaîne
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

    return rag_chain.invoke({"input": query})

if __name__ == "__main__":
    res = get_rag_response("Pose ta question ici")
    print(res["answer"])