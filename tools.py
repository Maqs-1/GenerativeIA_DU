from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain_google_genai import ChatGoogleGenerativeAI
# On importe ta fonction de RAG créée précédemment
# from rag_chain import get_rag_response 

def setup_tools():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # 1. Outil de recherche Web
    search = DuckDuckGoSearchRun()
    
    # 2. Outil de calcul
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    
    tools = [
        Tool(
            name="Recherche_Web",
            func=search.run,
            description="Utile pour répondre à des questions sur des événements actuels ou des informations en temps réel."
        ),
        Tool(
            name="Calculatrice",
            func=llm_math_chain.run,
            description="Utile pour effectuer des calculs mathématiques précis."
        ),
        # On ajoutera l'outil RAG ici une fois testé
    ]
    return tools