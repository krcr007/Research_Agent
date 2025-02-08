import streamlit as st
from typing import Optional
import os
import json
from phi.agent import Agent
from phi.tools.arxiv_toolkit import ArxivToolkit
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.wikipedia import WikipediaTools
from phi.model.openai import OpenAIChat
from phi.workflow import Workflow, RunResponse, RunEvent
from pydantic import BaseModel, Field

# Secure API Key Storage
st.title("Research Agent")

# User input for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

st.session_state.api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if not st.session_state.api_key:
    st.warning("Please enter your OpenAI API Key to proceed.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.session_state.api_key  # Store the API key securely

# Define Data Models
class Research(BaseModel):
    prompt: str = Field(..., description="Prompt to be given")

class Literature(BaseModel):
    url: str = Field(..., description="URL of the research paper")

class Keyinsights(BaseModel):
    url: str = Field(..., description="URL of the research paper")

# Define Agents
Arxiv_Agent = Agent(tools=[ArxivToolkit()], description="Search for research papers on Arxiv")
Wikipedia_Agent = Agent(tools=[WikipediaTools()], description="Search for details on Wikipedia")
WebSearch_Agent = Agent(tools=[DuckDuckGo()], description="Search for key insights online")

# Define Teams
Paper_team = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    team=[Arxiv_Agent, Wikipedia_Agent],
    instructions=[
        "Search for research papers on Arxiv.",
        "Retrieve additional details from Wikipedia."
    ],
    show_tool_calls=True,
    markdown=True
)

Literature_team = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    team=[Arxiv_Agent],
    instructions=[
        "Search for research papers on Arxiv.",
        "Create a structured summary table."
    ],
    show_tool_calls=True,
    markdown=True
)

Keyinsights_team = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    team=[Arxiv_Agent, Wikipedia_Agent, WebSearch_Agent],
    instructions=[
        "Retrieve relevant research papers from Arxiv.",
        "Extract key information from Wikipedia.",
        "Find latest insights via web search."
    ]
)

# Define Workflow
class Researcher(Workflow):
    def run(self, task: str, prompt: Optional[str] = None, url: Optional[str] = None,no:Optional[int]=1) -> RunResponse:
        if task == "research":
            return self.research(prompt)
        elif task == "literature":
            return self.literature(prompt,no)
        elif task == "keyinsights":
            return self.keyinsights(url)
        return RunResponse(content="Supported tasks: Research, Literature, Keyinsights.", event=RunEvent.workflow_completed)

    def research(self, prompt: Optional[str]) -> RunResponse:
        if not prompt:
            return RunResponse(content="Please provide a topic.", event=RunEvent.workflow_completed)
        response = Paper_team.run(message={"role": "user",
                                           "content": f"""Plz provide atleast 5 research papers for the topic : {prompt}. Also provide the link of the research papers at the end. Do no hallucinate at all always verify the paper and given the papres which are present."""})
        return RunResponse(content=response.content, event=RunEvent.workflow_completed)

    def literature(self, prompt: Optional[str],no:Optional[int]) -> RunResponse:
        if not prompt and no<=0:
            return RunResponse(content="Please provide a topic.", event=RunEvent.workflow_completed)
        response = Literature_team.run(message={"role": "user",
                                                "content": f"""The prompt given is the topic which is provided . The topic  is :{prompt}. 
                    First Find {no} research papers related to the topic and then Give the the following details in tabular format with columns:
        1. Title in IEEE Format for example (Journal Paper Reference:
        [1] A. Author, B. Author, and C. Author, "Title of the paper in double quotation marks," Journal Name in Italics, vol. X, no. Y, pp. xx–xx, Month, Year, doi: [DOI Number].

        Example:

        [1] J. Smith, R. Brown, and K. Lee, "Deep learning approaches for medical image analysis: A review," IEEE Transactions on Medical Imaging, vol. 39, no. 5, pp. 1234–1245, May 2020, doi: 10.1109/TMI.202)
        2. Methadolgy: Give a short descirpiton of that what is the paper about in 5-7 lines.
        3. Pros and cons of the paper.
        4. Year in which it was published.
        5. Journal in which it is published.

        Note:
        Never hallucinate and create things by your own mind.Always verify before giving the information."""})

        return RunResponse(content=response.content, event=RunEvent.workflow_completed)

    def keyinsights(self, url: Optional[str]) -> RunResponse:
        if not url:
            return RunResponse(content="Please provide a valid URL.", event=RunEvent.workflow_completed)
        response = Keyinsights_team.run(
            message={"role": "user", "content": f"""The url given to you is the research paper link. The url is :{url}.Provde with a in detail description of the papae what all things to be added in the description is:
        1. Methadology used.
        2. Algorithms or model used.
        3. Metric scores if any (Make sure you are legitimate about them and not create by your own mind)
        4. Links of the refreneces provided in the paper.
        Note:
        Never hallucinate and create things by your own mind.Always verify before giving the information"""})
        return RunResponse(content=response.content, event=RunEvent.workflow_completed)

# Streamlit App
st.title("Research Agent")

option = st.selectbox("Choose a task:", ["research", "literature", "keyinsights"])

prompt = None
url = None
no=1

if option == "research":
    prompt = st.text_input("Enter topic:")
elif option == "literature":
    prompt = st.text_input("Enter topic for literature review:")
    no=st.number_input("Enter the number of research papers you want for literature review:",format="%d",step=1,max_value=10)
elif option == "keyinsights":
    url = st.text_input("Enter research paper URL:")

if st.button("Run Task"):
    researcher_workflow = Researcher(session_id="research-agent-session")
    result = researcher_workflow.run(task=option, prompt=prompt, url=url,no=no)
    st.write(result.content)
