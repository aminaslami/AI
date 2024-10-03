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
# Generate the Test Case

from giskard import generate_testset

testset = generate_testset(
    knowledge_base,
    num_questions = 60,
    agent_description = "A chatbot answering questions about the Machine Learning School Website.",
    # This code is connect the GPT4 and generate all test cases
)

# ----------------------------------------------
# 6.1-Step
# Let's display a few samples from the test set.

test_set_df = testset.to_pandas()

for index, row in enumerate(test_set_df.head(3).iterrows()):
    print(f"Question {index + 1}: {row[1]['question']}")
    print(f"Reference answer: {row[1]['reference_answer']}")
    print("Reference context:")
    print(row[1]['reference_context'])
    print("*******************", end = "\n\n")

# ----------------------------------------------
# 6.2-Step
# Let's now save the test set to a file.

testset.save("test-set.jsonl")

# ----------------------------------------------
# 7-Step
# Prepare the Prompt Template

from langchain.prompts import PromptTemplate

template = """
Answeer the question based on the context below. If you can't
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
print(prompt.format(context = "Here is some context", question = "Here is a question"))

# ----------------------------------------------
# 8-Step

retriever = vectorstores.as_retriever()
# dir(retriever)
retriever.get_relevant_documents("What is the Machine Learning School")


# ----------------------------------------------
# 9-Step
# Create the RAG Chain

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

model = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model = MODEL)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | StrOutputParser()
)

# ----------------------------------------------
# 9.1-Step
# Let's make sure the chain works by testing it with a simple question

chain.invoke("{question}: What is the Machine Learning School")


# ----------------------------------------------
# 10-Step
# Evaluating the Model on the Test Set

def answer_fn(question, history = None):
    return chain.invoke({"question: question"})
    
    
# ----------------------------------------------
# 10.1-Step
# We can now use the evaluate() function to evaluate the model Test set.
# This function will compare the answer from chain with refernce answers in the test set.

from giskard.rag import evaluate
report = evaluate(answer_fn, testset = testset, knowledge_base = knowledge_base)

display(report)
   
# ----------------------------------------------
# 10.1-Step
