import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

def get_rag_chain():
    # Utilisation des embeddings Google (plus stable dans ton venv actuel)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Chargement du vectorstore existant (Chroma)
    vectorstore = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    
    # Configuration du modèle
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Ajout d'une mémoire pour que le RAG se souvienne du contexte
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Création de la chaîne (inspirée de rag_langchain.py qui fonctionne)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return chain

# Fonction pour l'agent
def query_rag(question):
    chain = get_rag_chain()
    result = chain.invoke({"question": question})
    return result["answer"]