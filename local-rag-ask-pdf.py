#Step1
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
MODEL = "mixtral:8x7b"
MODEL = "llama2"

"""
requirement librires

pip instal python_dotenv
"""

#Step2-------------------------------------------------------
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

if MODEL.startswith("gpt"):
    model = ChatOpenAI(api_key = OPENAI_API_KEY, model=MODEL)
    embeddings = OpenAIEmbeddings()
else:
    model = Ollama(model=MODEL)
    embeddings = OllamaEmbeddings()
    
model.invoke("Tell me a joke")
"""
pip install langchain_openai
pip install langchain_community
"""

#Step3-------------------------------------------------------
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = model | parser
chain.invoke("Tell me a joke")

 

#Step4-------------------------------------------------------
from langchain_community.document_loaders import PyPDFLoader

# Go the (website) and download any ML pdf topic
loader = PyPDFLoader("mlschool.pdf")
pages = loader.load_and_split()
pages

"""
pip install pypdf
"""

#Step5-------------------------------------------------------
from langchain.propmts import PromptTemplate

template = """
Answer the question based on the context below. If you can't
answer the question, reply "I don't know".

Context: {context}

Question" {question}
"""

prompt = PromptTemplate.from_template(template)
print(prompt.format(context="Here is some context", question="Here is a question"))


#Step6-------------------------------------------------------
chain = prompt | model | parser

#Step7-------------------------------------------------------
chain.input_schema.schema()

#Step8-------------------------------------------------------
chain.invoke(
    {
        "context": "The name I was given was Mahdi",
        "question": "What's my name?"
        # 'Your name is Mahdi.'
    }
)

#Step9-------------------------------------------------------
from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstores = DocArrayInMemorySearch.from_documents(
    pages, 
    embedding=embedding

)

# pip install docarray
# pip install pydantic==1.10.8

#Step10-------------------------------------------------------
retriever = vectorstores.as_retriever()

retriever.invoke("Machine learning")

#Step11-------------------------------------------------------
from operator import itemgetter

chain = 

{
    {
    "context": itemgetter("question") | retriever, 
    "question": itemgetter("question")
    }
    | prompt
    | model
    | parser
}

chain.invoke({"question": "What is machine learning?"})

#Step5-------------------------------------------------------
question = [
    "What is the purpose of the course?",
    "How many hours of live sessions?",
    "How many coding assignments are there in the program?",
    "Is there a program certificate upon completion",
    "What programing language will be used in the program",
    "How much does the program cost",
]

#Step5-------------------------------------------------------


#Step5-------------------------------------------------------


#Step5-------------------------------------------------------


#Step5-------------------------------------------------------


#Step5-------------------------------------------------------


#Step5-------------------------------------------------------



#Step5-------------------------------------------------------