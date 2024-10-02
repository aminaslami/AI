# 1-Step - The code is writed on Vscode application
# How to evaluate a RAG Application:

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"

# ------------------------------------------
# 2-Step
# Scrape the Webiste and Split the Content

from langchain_community.doucment_loaders import WebBaseLoader
from langchain.text_spliter import RecursiveCharacterTextSpliter

text_spliter = RecursiveCharacterTextSpliter(chunk_size = 1000, chunk_overlap = 20)

loader = WebBaseLoader("https://www.lm.school/")
documents = loader.load_and_split(text_spliter)
documents

# ------------------------------------------
# 3-Step
# How many documents is loaded

len(documents)

# ------------------------------------------
# 4-Step
# Load the Content in a Vector Store

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstores = DocArrayInMemorySearch.from_documents(
    documents, 
    embeddings=OpenAIEmbeddings()
)

# ------------------------------------------
# 5-Step
# Create a Knowledge Base

import pandas as pd

df = pd.DataFrame([d.page_content for d in documents], columns["text"])
df.head(10)

# ------------------------------------------
# 5.1-Step
# We can now create a Knowlege Base using the DataFrame we created before.

from giskard.rag import KnowledgeBase

knowledge_base = KnowledgeBase(df)

# ------------------------------------------
# 6-Step
# 




