import os
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import LLMMathChain
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_chain import query_rag # Import de ta fonction RAG

def setup_tools():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # 1. Recherche Web haute performance
    search = TavilySearchResults(k=3)
    
    # 2. Calculatrice (via LLMMathChain comme dans tes docs)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    
    tools = [
        Tool(
            name="Documentation_Interne",
            func=query_rag,
            description="À UTILISER EN PRIORITÉ pour les Incoterms 2020, rapports CMA CGM et DHL."
        ),
        Tool(
            name="Recherche_Web",
            func=search.run,
            description="Utile pour la météo, le prix du baril ou les actualités logistiques."
        ),
        Tool(
            name="Calculatrice",
            func=llm_math_chain.run,
            description="Utile pour les calculs de coûts, de taxes ou de volumes."
        )
    ]
    return tools