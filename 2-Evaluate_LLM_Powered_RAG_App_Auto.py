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

