import streamlit as st

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.sql_database import SQLDatabase

uri = 'clickhouse+native://localhost/ecommerce'
db = SQLDatabase.from_uri(uri)

llm = ChatOpenAI(temperature=0.1,
                 model='gpt-4-1106-preview',
                 streaming=True)

toolkit = SQLDatabaseToolkit(db=db,
                             llm=llm,)


agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS
)

st.header("Gen AI for Data Analysis : SQL DB!")
st.write("Example: How many active customers from the United Kingdom do we have?")
request = st.text_area('Enter your query! ', height=100)
submit = st.button("submit", type="primary")

if submit and request:
    response = agent_executor.run(request)
    st.write(response)
