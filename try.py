import os
import numpy as np
import gradio as gr
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import logging
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import requests
from bs4 import BeautifulSoup
import re

# Fix for OpenMP multiple runtimes error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize Gemini models
model_chat = genai.GenerativeModel("gemini-pro")
chat = model_chat.start_chat(history=[])

# Update model to the latest version
model_vision = genai.GenerativeModel('gemini-1.5-flash')

# Load the Pegasus tokenizer and model for summarization
def load_summarizer():
    model_name = "google/pegasus-large"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_summarizer()

# Define function to handle questions
def handle_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)

    # Summarize context
    context = "\n".join([doc.page_content for doc in docs])
    summarized_context = summarize_text(context)

    # Create conversational chain
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question, "context": summarized_context}, return_only_outputs=True)
    
    return response["output_text"]

# Create Gradio Interface
iface = gr.Interface(
    fn=handle_question,
    inputs=gr.inputs.Textbox(label="Ask a Question"),
    outputs=gr.outputs.Textbox(label="Response"),
    title="Gemini Q&A Application",
    description="Ask your questions related to the PDF documents and get responses powered by Gemini AI."
)

# Launch the Gradio interface
iface.launch()
