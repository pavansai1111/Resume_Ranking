import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text.strip()

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()
    return cosine_similarities

# Streamlit UI
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# Full-width image banner (showing only half of the image)
st.markdown(
    """
    <style>
        .banner-container {
            width: 100%;
            height: 50vh; /* Display only half the image */
            overflow: hidden;
            display: flex;
            justify-content: center;
        }
        .banner-img {
            width: 100%;
            height: auto;
        }
    </style>
    <div class='banner-container'>
        <img src='https://www.theforage.com/blog/wp-content/uploads/2022/09/Depositphotos_95377176_L.jpg' class='banner-img'>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='font-size:50px; text-align:center;'>📄 AI Resume Screening & Candidate Ranking</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='font-size:40px;'>🔍 Job Description</h2>", unsafe_allow_html=True)
    job_description = st.text_area("Enter the job description", height=500, key="job_description")
    st.markdown("<h2 style='font-size:40px;'>📂 Upload Resumes</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Main content
st.markdown("---")
st.markdown("<h2 style='font-size:40px;'>ℹ️ How to Use</h2>", unsafe_allow_html=True)
st.markdown("<div style='font-size:30px;'>1. Enter the job description in the sidebar.</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size:30px;'>2. Upload one or multiple PDF resumes from the sidebar.</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size:30px;'>3. The system will rank resumes based on relevance.</div>", unsafe_allow_html=True)
st.markdown("<div style='font-size:30px;'>4. View results below after processing.</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h2 style='font-size:40px;'>📊 About</h2>", unsafe_allow_html=True)
st.markdown("<div style='font-size:30px;'>This tool uses AI to analyze and rank resumes based on the provided job description using TF-IDF and cosine similarity.</div>", unsafe_allow_html=True)

if uploaded_files and job_description:
    with st.spinner("Processing resumes..."):
        resumes = [extract_text_from_pdf(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes)
        results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
    
    st.success("✅ Ranking Complete!")
    st.markdown("<h2 style='font-size:40px;'>🏆 Ranked Resumes</h2>", unsafe_allow_html=True)
    st.dataframe(results.style.format({"Score": "{:.2f}"}).bar(subset=["Score"], color='#4CAF50'))
