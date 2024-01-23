import requests
import io
import pandas as pd
from langchain.agents import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional
from langchain.chat_models import ChatOpenAI
import langchain

load_dotenv()

CH_HOST = 'http://localhost:8123'
pd.set_option('display.max_colwidth', 1000)


def get_clickhouse_data(query, host=CH_HOST, connection_timeout=1500):
    r = requests.post(host,
                      params={'query': query},
                      timeout=connection_timeout)

    return r.text


def get_clickhouse_df(query, host=CH_HOST, connection_timeout=1500):
    data = get_clickhouse_data(query, host, connection_timeout)
    df = pd.read_csv(io.StringIO(data), sep='\t')
    return df


class SQLQuery(BaseModel):
    query: str = Field(description="SQL query to execute")


@tool(args_schema=SQLQuery)
def execute_sql(query: str) -> str:
    """Returns the result of SQL query execution"""
    return get_clickhouse_data(query)


class SQLTable(BaseModel):
    database: str = Field(description="Database name")
    table: str = Field(description="Table name")


@tool(args_schema=SQLTable)
def get_table_columns(database: str, table: str) -> str:
    """Returns list of table column names and types in JSON"""

    q = '''
    select name, type
    from system.columns 
    where database = '{database}'
        and table = '{table}'
    format TabSeparatedWithNames
    '''.format(database=database, table=table)

    return str(get_clickhouse_df(q).to_dict('records'))


print(get_table_columns({'database': 'ecommerce', 'table': 'sessions'}))


class SQLTableColumn(BaseModel):
    database: str = Field(description="Database name")
    table: str = Field(description="Table name")
    column: str = Field(description="Column name")
    n: Optional[int] = Field(description="Number of rows, default limit 10")


@tool(args_schema=SQLTableColumn)
def get_table_column_distr(database: str, table: str, column: str, n: int = 10) -> str:
    """Returns top n values for the column in JSON"""

    q = '''
    select {column}, count(1) as count
    from {database}.{table} 
    group by 1
    order by 2 desc 
    limit {n}
    format TabSeparatedWithNames
    '''.format(database=database, table=table, column=column, n=n)

    return str(list(get_clickhouse_df(q)[column].values))


print(get_table_column_distr({'database': 'ecommerce', 'table': 'sessions', 'column': 'os'}))

from langchain.tools.render import format_tool_to_openai_function

sql_functions = list(map(format_tool_to_openai_function, [execute_sql, get_table_columns, get_table_column_distr]))

sql_tools = {
    'execute_sql': execute_sql,
    'get_table_columns': get_table_columns,
    'get_table_column_distr': get_table_column_distr
}

llm = (ChatOpenAI(temperature=0.1,
                  model='gpt-4-1106-preview')
       .bind(functions=sql_functions))

system_message = '''You are working as a product analyst for the e-commerce company. Your work is very important, 
since your product team makes decisions based on the data you provide. So, you are extremely accurate with the 
numbers you provided. If you're not sure about the details of the request, you don't provide the answer and ask 
follow-up questions to have a clear understanding. You are very helpful and try your best to answer the questions.

All the data is stored in SQL Database. Here is the list of tables (in the format <database>.<table>) with descriptions:
- ecommerce.users - information about the customers, one row - one customer
- ecommerce.sessions - information about the sessions customers made on our web site, one row - one session
'''

langchain.debug = True

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.schema import SystemMessage

agent_kwargs = {
    "system_message": SystemMessage(content=system_message)
}

analyst_agent_openai = initialize_agent(
    llm=ChatOpenAI(temperature=0.1,
                   model='gpt-4-1106-preview'),
    agent=AgentType.OPENAI_FUNCTIONS,
    tools=[execute_sql, get_table_columns, get_table_column_distr],
    agent_kwargs=agent_kwargs,
    verbose=True,
    max_iterations=10,
    early_stopping_method='generate'
)

analyst_agent_openai.get_input_schema().schema()
analyst_agent_openai.get_output_schema().schema()

analyst_agent_openai.run("How many active customers from the United Kingdom do we have?")

# ReAct agent
agent_kwargs = {
    "prefix": system_message
}

analyst_agent_react = initialize_agent(
    llm=ChatOpenAI(temperature=0.1,
                   model='gpt-4-1106-preview'),
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    tools=[execute_sql, get_table_columns, get_table_column_distr],
    agent_kwargs=agent_kwargs,
    verbose=True,
    max_iterations=10,
    early_stopping_method='generate'
)

for message in analyst_agent_react.agent.llm_chain.prompt.messages:
    print(message.prompt.template)

analyst_agent_react.run("How many active customers from the United Kingdom do we have?")
