import streamlit as st

from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

load_dotenv()

df = pd.read_csv("data/telco-customer-churn.csv")

llm = ChatOpenAI(temperature=0.1,
                 model='gpt-4-1106-preview',
                 streaming=True)

agent = create_pandas_dataframe_agent(llm,
                                      df,
                                      verbose=True, )

st.header("Gen AI for Data Analysis. Pandas DF !")
request = st.text_area('Enter your query! ', height=100)
submit = st.button("submit", type="primary")

if submit and request:
    response = agent.run(request)
    st.write(response)
