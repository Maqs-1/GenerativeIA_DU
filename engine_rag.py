from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

# Charge les variables d'environnement (.env)
load_dotenv()

def build_vectorstore():
    print("1. Chargement des documents depuis le dossier 'data'...")
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    
    if not docs:
        print("ERREUR : Aucun document trouvé. Placez vos PDF dans 'data/'.")
        return None

    print(f"2. Découpage des {len(docs)} pages trouvées...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    print("3. Vectorisation et création de la base via Gemini...")
    # Utilisation du modèle d'embedding officiel de Google
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./vectorstore"
    )
    
    print("SUCCÈS ! Base vectorielle générée avec Gemini dans './vectorstore'.")
    return vectorstore

if __name__ == "__main__":
    build_vectorstore()