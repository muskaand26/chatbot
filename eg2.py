import os
import numpy as np
import streamlit as st
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
from PIL import Image
import requests
from bs4 import BeautifulSoup
import asyncio
import re

# Fix for OpenMP multiple runtimes error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Set Streamlit page configuration
st.set_page_config(page_title="Gemini Q&A and Chat Application", page_icon="🤖", layout="wide")



# Initialize session state for light/dark mode
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Default to light mode

# Function to toggle theme
def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# Show the toggle button with emojis in the sidebar
if st.session_state.theme == 'light':
    emoji = "🌙"  # Sun emoji for light mode
else:
    emoji = "☀️"  # Moon emoji for dark mode

st.sidebar.button(emoji, on_click=toggle_theme)  # Emoji button in sidebar



# Inject JavaScript to handle theme changes and toggle emoji
if st.session_state.theme == 'light':
    theme_class = ""
else:
    theme_class = "dark-theme"

# Toggle button and theme logic in JavaScript
st.markdown(f"""
    <style>
        body.dark-theme {{
            background-color: #333;
            color: white;
        }}
    </style>

    <script>
        // Get references to the body
        const body = document.body;

        // Apply initial theme from session state
        if ("{theme_class}" === "dark-theme") {{
            body.classList.add("dark-theme");
        }}

        // Handle theme toggle and send the new theme to Streamlit session state
        document.querySelector('button[aria-label="{emoji}"]').addEventListener("click", function() {{
            if (body.classList.contains("dark-theme")) {{
                body.classList.remove("dark-theme");
                Streamlit.setComponentValue({{"theme": "light"}});
            }} else {{
                body.classList.add("dark-theme");
                Streamlit.setComponentValue({{"theme": "dark"}});
            }}
        }});
    </script>
""", unsafe_allow_html=True)

# Inject CSS for dark and light modes based on theme
if st.session_state.theme == 'dark':
    st.markdown("""
        <style>
            /* Dark mode settings */
            body {
                background-color: black;
                color: red;
            }
            [data-testid="stAppViewContainer"] {
                background-color: black;
            }
            [data-testid="stSidebar"] {
                background-color: black;
                color: red;
            }
            a { color: red !important; }  /* Links */
            .stTextInput > div > div > input {
                color: red !important;
                background-color: transparent; /* Make input transparent */
                border: 1px solid red; /* Red border for inputs */
            }
            /* Ensure all text in the sidebar is red in dark mode */
            .css-1v3fvcr, .css-qbe2hs, .css-1v0mbdj, .stSelectbox, .stTextArea, .stButton {
                color: red !important;
            }
            div[role="button"], h1, h2, h3, h4, h5, h6, p, label {
                color: orange !important;
            }
            header, .stApp {
                background-color: black !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
            /* Light mode settings */
            body {
                background-color: white;
                color: black;
            }
            [data-testid="stAppViewContainer"] {
                background-color: white;
            }
            [data-testid="stSidebar"] {
                background-color: white;
                color: black;
            }
            a { color: black !important; }  /* Links */
            .stTextInput > div > div > input {
                color: black !important;
                background-color: transparent; /* Make input transparent */
                border: 1px solid black; /* Black border for inputs */
            }
            /* Ensure all text is black in light mode */
            .css-1v3fvcr, .css-qbe2hs, .css-1v0mbdj, .stSelectbox, .stTextArea, .stButton {
                color: black !important;
            }
            div[role="button"], h1, h2, h3, h4, h5, h6, p, label {
                color: black !important;
            }
            header, .stApp {
                background-color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize Gemini models
model_chat = genai.GenerativeModel("gemini-pro")
chat = model_chat.start_chat(history=[])

# Update model to the latest version
model_vision = genai.GenerativeModel('gemini-1.5-flash')

# Store scraped content globally
scraped_content = ""


# Async function to get responses from the Gemini Pro model
async def get_gemini_response(question, context=None):
    if context:
        question = f"{question}\n\nContext:\n{context}"
    response = chat.send_message(question, stream=True)
    return response

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up the conversational chain for PDF Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available, just say "Answer is not available in the context."\n\n
    Context:\n{context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Intelligent Guard Rails based on context
def intelligent_guard_rails(text):
    # Define inappropriate keywords
    inappropriate_keywords = ['violence', 'sexual', 'politics', 'inappropriate']
    # Define allowed contexts
    allowed_contexts = ['educational', 'workplace discussion', 'policy debate']

    # Check for inappropriate keywords and context
    if any(keyword in text.lower() for keyword in inappropriate_keywords):
        if not any(context in text.lower() for context in allowed_contexts):
            return "Inappropriate content blocked by guard rails."
    return text

# Function to format chatbot responses
def format_response(response):
    response = re.sub(r'\*+', '', response)  # Remove any number of consecutive '*'
    response = re.sub(r'#{1,6} (.*)', r'**\1**', response)  # Headings (you can keep this if needed)
    response = re.sub(r'\n\s*\n', '\n\n', response)  # Proper line breaks
    return response

# Load the Pegasus tokenizer and model for summarization
@st.cache_resource
def load_summarizer():
    model_name = "google/pegasus-large"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_summarizer()

# Function to summarize text
def summarize_text(text):
    cleaned_text = text.replace("\n", " ").replace("\r", " ")
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=100, 
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Web Scraping Function for HR Policies & IT Support Content
def scrape_web_content(url):
    global scraped_content
    try:
        response = requests.get(url, verify=False)  # Bypass SSL verification
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        scraped_content = "\n".join([para.get_text() for para in paragraphs])
        inappropriate_keywords = ['politics', 'sexual', 'violence', 'inappropriate']
        if any(keyword in scraped_content.lower() for keyword in inappropriate_keywords):
            scraped_content = "The content fetched contains inappropriate information, which is not allowed."
        return scraped_content
    except Exception as e:
        return f"Failed to scrape the content: {str(e)}"

# Function to get Gemini response for text and image
def get_gemini_response_image(input_text, image):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text

# Async function to get responses from the Gemini Pro model
async def get_gemini_response(question, context=None):
    if context:
        question = f"{question}\n\nContext:\n{context}"
    response = chat.send_message(question, stream=True)
    return response


def handle_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)
    
    # Implementing RAGAS - summarization of the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])
    summarized_context = summarize_text(context)

    # Generate the answer using the summarized context
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question, "context": summarized_context}, return_only_outputs=True)
    
    filtered_response = format_response(response["output_text"])
    return filtered_response


from sklearn.feature_extraction.text import CountVectorizer

# Function to log metrics (in terminal) and calculate more RAG metrics
def calculate_rag_metrics(user_question, docs, summarized_context):
    if not docs:
        # Handle the case where no documents are retrieved
        logging.info("No documents were retrieved for the question.")
        return 0, 0, 0, 0, 0, 0, 0

    # More flexible detection of relevance by using word overlap
    vectorizer = CountVectorizer().fit([user_question])
    user_question_tokens = set(vectorizer.get_feature_names_out())

    relevant_docs = []
    for doc in docs:
        doc_tokens = set(vectorizer.fit([doc.page_content]).get_feature_names_out())
        overlap = user_question_tokens.intersection(doc_tokens)
        if overlap:  # If there's any word overlap, consider it relevant
            relevant_docs.append(doc)

    # Calculate metrics
    # Accuracy: Proportion of relevant documents retrieved out of total retrieved documents
    accuracy = len(relevant_docs) / len(docs) if docs else 0

    # Precision: Proportion of retrieved docs that are relevant
    precision = len(relevant_docs) / len(docs) if docs else 0

    # Recall: We assume the number of relevant docs in `docs` is the total
    recall = len(relevant_docs) / len(docs) if docs else 0

    # F1 Score: Harmonic mean of precision and recall
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Faithfulness: Proportion of retrieved documents that faithfully contain the user's question context
    faithfulness = np.mean([len(user_question_tokens.intersection(set(vectorizer.fit([doc.page_content]).get_feature_names_out()))) > 0 for doc in docs])

    # Relevance: Assuming relevance is based on overlap or predefined metadata
    relevance = np.mean([1 if len(overlap) > 0 else 0 for doc in docs])

    # Diversity: Calculate diversity based on unique page numbers or sections
    unique_pages = len(set([doc.metadata.get('page_number', None) for doc in docs if 'page_number' in doc.metadata]))
    diversity = unique_pages / len(docs) if docs else 0

    # Log the metrics to the terminal
    logging.info(f"RAG Model Accuracy: {accuracy*100:.2f}%")
    logging.info(f"Summarization Length: {len(summarized_context)} characters")
    logging.info(f"Precision: {precision:.2f}")
    logging.info(f"Recall: {recall:.2f}")
    logging.info(f"F1 Score: {f1_score:.2f}")
    logging.info(f"Faithfulness: {faithfulness:.2f}")
    logging.info(f"Relevance: {relevance:.2f}")
    logging.info(f"Diversity: {diversity:.2f}")

    return accuracy, precision, recall, f1_score, faithfulness, relevance, diversity



# Function to handle RAG Question and display metrics
def handle_rag_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Increase the number of relevant documents fetched to improve accuracy
    docs = new_db.similarity_search(user_question, k=10)  # Fetch 10 instead of 5

    # Generate answers from retrieved documents
    answers = []
    for doc in docs:
        chain = get_conversational_chain()
        answer = chain({"input_documents": [doc], "question": user_question}, return_only_outputs=True)
        answers.append(answer["output_text"])

    unique_answers = list(set(answers))
    filtered_answers = [format_response(answer) for answer in unique_answers if answer != "Answer is not available in the context."]

    summarized_context = summarize_text("\n".join([doc.page_content for doc in docs]))
    
    # Calculate and log metrics (hidden from UI)
    accuracy, context_precision, faithfulness = calculate_rag_metrics(user_question, docs, summarized_context)

    # Optionally, improve filtering logic based on metrics
    filtered_answers = [answer for answer in filtered_answers if faithfulness > 0.5]  # Only show faithful answers

    return filtered_answers

# Define the FAQ section
faq_content = {
    "What is GAIL?": "GAIL (India) Limited is the largest state-owned natural gas processing and distribution company in India.",
    "How to apply for a job in GAIL?": "You can visit the GAIL careers website and apply for open positions.",
    # Add more FAQ entries as needed
}



# FAQ Section in Sidebar
st.sidebar.title("FAQ Section (GAIL)")
for question, answer in faq_content.items():
    with st.sidebar.expander(question):
        st.write(answer)
# Define the chain variable
chain = get_conversational_chain()


@st.cache_resource
def load_summarizer():
    model_name = "google/pegasus-large"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_summarizer()

# Function to summarize text
def summarize_text(text):
    cleaned_text = text.replace("\n", " ").replace("\r", " ")
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=100, 
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Sidebar for web scraping and chat history
st.sidebar.title("Web Scraping & Chat History")

# Web Scraping section
st.sidebar.title("HR & IT Policies Scraping")
scraping_url = st.sidebar.text_input("Enter URL to scrape HR or IT content:")
if st.sidebar.button("Scrape Content"):
    if scraping_url:
        with st.spinner("Scraping content..."):
            scraped_content = scrape_web_content(scraping_url)
            st.sidebar.write("Scraped Content:")
            st.sidebar.write(scraped_content)

# Chat History section
st.sidebar.title("Chat History")
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Display chat history from all tabs
with st.sidebar.expander("View Chat History"):
    for role, text in st.session_state['chat_history']:
        st.write(f"{role}: {text}")


# Tabs for different functionalities
tabs = st.tabs(["Chat with AI", "PDF Q&A", "Image Analysis"])

# Chat with AI tab
with tabs[0]:
    st.subheader("Chat with Gemini Pro AI")
    input_text = st.text_input("Enter your message:", key="input")
    
    if st.button("Send"):
        if input_text:
            # Use scraped content in the response if relevant
            if scraped_content and 'scraped' in input_text.lower():
                response = asyncio.run(get_gemini_response(input_text, scraped_content))
            else:
                response = asyncio.run(get_gemini_response(input_text))
            
            st.session_state['chat_history'].append(("You", input_text))
            formatted_response = ""
            for chunk in response:
                formatted_response += chunk.text
            formatted_response = format_response(formatted_response)
            st.markdown(formatted_response)
            st.session_state['chat_history'].append(("Bot", formatted_response))

# PDF Q&A Tab
with tabs[1]:
    st.subheader("Ask Questions or Summarize PDF Content")
    
    # PDF File Uploader
    pdf_docs = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    
    # Avoid multiple processing by ensuring pdf_docs is processed once
    if pdf_docs and "processed" not in st.session_state:
        with st.spinner("Processing PDFs..."):
            # Extract text from uploaded PDFs (without showing full content)
            pdf_text = get_pdf_text(pdf_docs)
            
            # Convert extracted text into text chunks and store in vector DB
            chunks = get_text_chunks(pdf_text)
            get_vector_store(chunks)
            st.success("PDFs Processed and Text Stored!")
            st.session_state["processed"] = True  # Mark the PDFs as processed

    # Two buttons: One for asking questions, one for summary
    col1, col2 = st.columns(2)
    
    # Button 1: For asking a question
    with col1:
        user_question = st.text_input("Ask a question about the uploaded PDFs:")
        
        if user_question and pdf_docs:
            # Fetch the answer based on user question
            with st.spinner("Fetching answer..."):
                pdf_answer = handle_user_question(user_question)
                st.write("Answer:")
                st.write(pdf_answer)

            # Log metrics for question-answering in the terminal
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question, k=5)
            summarized_context = summarize_text("\n".join([doc.page_content for doc in docs]))
            
            # Calculate and log improved RAG metrics
            accuracy, precision, recall, faithfulness, f1, relevance, diversity = calculate_rag_metrics(user_question, docs, summarized_context)
            print(f"Metrics Logged in Terminal: \nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nFaithfulness: {faithfulness:.2f}\nF1 Score: {f1:.2f}\nRelevance: {relevance:.2f}\nDiversity: {diversity:.2f}")

    # Button 2: For summarizing the content directly (without question)
    with col2:
        if st.button("Show Summary of PDF"):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search("", k=5)  # No question provided, so summarize the overall content
            
            # Summarize the entire document context
            summarized_context = summarize_text("\n".join([doc.page_content for doc in docs]))
            st.write("Summary of PDF Content:")
            st.write(summarized_context)



# Image Analysis tab
with tabs[2]:
    st.subheader("Analyze Images")
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["jpg", "png", "jpeg"])
    image_analysis_text = st.text_input("Enter a prompt related to the image:")
    
    if st.button("Analyze Image"):
        if uploaded_image:
            st.image(uploaded_image)
            if image_analysis_text:
                image = Image.open(uploaded_image)
                response = get_gemini_response_image(image_analysis_text, image)
                st.write("Analysis Report: ")
                st.write(response)
