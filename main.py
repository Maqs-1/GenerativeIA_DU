import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilyAnswer
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Import de ton pipeline RAG
from rag_chain import get_rag_response

load_dotenv()

# --- 1. OUTIL RAG (Tes PDF) ---
@tool
def recherche_documents_internes(query: str) -> str:
    """
    Consulte les documents internes de l'entreprise (Incoterms 2020, rapports CMA CGM, DHL).
    À utiliser pour toute question sur les procédures, contrats ou rapports RSE.
    """
    resultat = get_rag_response(query)
    return resultat["answer"] if isinstance(resultat, dict) else resultat

# --- 2. OUTIL CALCULATRICE ---
@tool
def calculatrice(expression: str) -> str:
    """
    Utile pour effectuer des calculs mathématiques complexes (coûts logistiques, 
    volumes de conteneurs, taxes douanières). L'expression doit être mathématique (ex: '500 * 1.2').
    """
    try:
        # Utilisation de eval sécurisé pour les calculs simples
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception:
        return "Erreur de calcul. Assurez-vous d'envoyer une expression mathématique valide."

# --- 3. OUTIL MÉTÉO & RECHERCHE (Via Tavily) ---
# On configure Tavily pour qu'il puisse aussi servir de moteur météo en temps réel
search_tool = TavilyAnswer(max_results=3)

@tool
def meteo_portuaire(ville: str) -> str:
    """
    Récupère la météo en temps réel pour une ville ou un port spécifique. 
    Utile pour prévoir des retards de navigation ou de transport.
    """
    return search_tool.run(f"Météo actuelle et prévisions pour {ville}")

# --- CONFIGURATION DE L'AGENT ---

tools = [recherche_documents_internes, calculatrice, search_tool, meteo_portuaire]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un assistant logistique expert. 
    1. Utilise 'recherche_documents_internes' pour les Incoterms ou rapports d'entreprise.
    2. Utilise 'calculatrice' pour tout calcul de coût ou de volume.
    3. Utilise 'meteo_portuaire' pour les conditions climatiques.
    4. Utilise 'tavily_search_results_json' pour les actualités mondiales du transport."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- TESTS ---
if __name__ == "__main__":
    # Test Calcul + Météo
    print("\n--- Test Combiné ---")
    question = "Quelle est la météo au port de Rotterdam et quel serait le coût total de 50 conteneurs si l'unité coûte 1200€ ?"
    agent_executor.invoke({"input": question})