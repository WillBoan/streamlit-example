from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Contend Legal Demo

In the meantime, below is an example of what you can do with just a few lines of code:
"""


import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredHTMLLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.vectorstores import Chroma

OPENAI_API_KEY="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"


REACT_QUESTION = """
I've received a Section 21 eviction notice. 
I live in England.
The date today is Oct 6, 2023.
Is the eviction notice valid?
"""

loader = DirectoryLoader('data/citizens-advice/', glob="**/*.txt", show_progress=True, loader_cls=TextLoader)
loaded_docs = loader.load()

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
db = Chroma.from_documents(loaded_docs, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))


@tool("search docs")
def search_docs(query: str) -> str:
    """Searches the database for legal information."""
    docs = db.similarity_search_with_relevance_scores(query, k=1)
    return docs[0][0].page_content


@tool("get user input")
def get_user_input(query: str) -> str:
    """Use this to ask the user a single question about their situation."""
    user_input = input()
    return user_input


tools = [search_docs, get_user_input]

chat_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
react = initialize_agent(
    tools,
    chat_model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=False,
    max_iterations=8
)



st.title(' Contend Legal Demo ')
input_text = st.text_input('Enter Your Text: ')

if input_text:
    
    # title = chainT.run(input_text)
    # wikipedia_research = wikipedia.run(input_text) 
    # script = chainS.run(title=title, wikipedia_research=wikipedia_research)
 
    st.write(react.run(REACT_QUESTION3)) 
    # st.write(script) 
 
    # with st.expander('Wikipedia-based exploration: '): 
    #     st.info(wikipedia_research)


# with st.echo(code_location='below'):
#     total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#     num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

#     Point = namedtuple('Point', 'x y')
#     data = []

#     points_per_turn = total_points / num_turns

#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))

#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q'))
