# Dans ton main.py révisé
from tools import setup_tools
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
tools = setup_tools()

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, # Le plus stable
    verbose=True,
    handle_parsing_errors=True # Crucial pour éviter les plantages
)