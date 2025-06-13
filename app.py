import streamlit as st
from pdfminer.high_level import extract_text
import tempfile
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from google.genai import types
import numpy as np
import re
import os
from dotenv import load_dotenv
from fpdf import FPDF
import base64

load_dotenv()

# Configure Google Gemini
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key) 

# PDF Generation Setup
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Optimized CV', 0, 1, 'C')
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_download_link(pdf_output, filename):
    """Generate a download link for PDF"""
    b64 = base64.b64encode(pdf_output).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download Optimized CV as PDF</a>'
    return href

st.title("AI CV Optimizer")

# Upload CV
uploaded_file = st.file_uploader("Upload your CV (PDF only)", type=["pdf"])

# Job Description Input
st.header("Enter Job Description")
jd_text = st.text_area("Paste the Job Description here", height=300)

# Store optimized sections globally
if 'optimized_sections' not in st.session_state:
    st.session_state.optimized_sections = []

def preprocess_text(text):
    """Clean and split text into meaningful chunks"""
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = re.split(r'\n\s*\n', text)
    return [chunk for chunk in chunks if len(chunk) > 30]

def get_embedding(text):
    """Get embedding using Google's embedding model"""
    try:
        result = client.models.embed_content(
            model="models/text-embedding-004",
            contents=text if isinstance(text, list) else [text],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        if isinstance(text, list):
            return [e.values for e in result.embeddings]
        return result.embeddings[0].values
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

def find_relevant_sections(cv_chunks, jd_embedding, top_n=3):
    """Find most relevant CV sections to job description"""
    if not cv_chunks or jd_embedding is None:
        return []
    
    chunk_embeddings = get_embedding(cv_chunks)
    if not chunk_embeddings:
        return []
    
    similarities = []
    jd_2d = np.array(jd_embedding).reshape(1, -1)
    
    for chunk, chunk_emb in zip(cv_chunks, chunk_embeddings):
        chunk_2d = np.array(chunk_emb).reshape(1, -1)
        sim = cosine_similarity(jd_2d, chunk_2d)[0][0]
        similarities.append((chunk, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, sim in similarities[:top_n]]

def calculate_match_score(cv_chunks, jd_embedding):
    """Calculate overall match percentage between CV and JD"""
    if not cv_chunks or jd_embedding is None:
        return 0
    
    chunk_embeddings = get_embedding(cv_chunks)
    if not chunk_embeddings:
        return 0
    
    jd_2d = np.array(jd_embedding).reshape(1, -1)
    similarities = []
    
    for chunk_emb in chunk_embeddings:
        chunk_2d = np.array(chunk_emb).reshape(1, -1)
        sim = cosine_similarity(jd_2d, chunk_2d)[0][0]
        similarities.append(sim)
    
    if not similarities:
        return 0
    
    total_length = sum(len(chunk) for chunk in cv_chunks)
    weighted_sum = sum(sim * (len(chunk)/total_length) for sim, chunk in zip(similarities, cv_chunks))
    return min(100, max(0, weighted_sum * 100))

def display_match_score(score):
    """Visual display of match percentage"""
    st.subheader("2CV-Job Match Score")
    color = f"hsl({score * 1.2}, 100%, 45%)"
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #f0f0f0 {score}%, transparent {score}%);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 2px solid {color};
        ">
            <h2 style="color: {color}; margin: 0;">{score:.0f}% Match</h2>
            <p style="margin: 5px 0 0;">with the job description</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.expander("What does this score mean?"):
        st.markdown("""
        - **80-100%**: Excellent match
        - **60-79%**: Good match
        - **40-59%**: Partial match
        - **0-39%**: Weak match
        """)

def optimize_section(section, jd_text):
    """Use Gemini to optimize a CV section based on JD"""
    prompt = f"""
    Rewrite this CV section to better match the job description below. 
    Keep it professional and truthful, but incorporate relevant keywords naturally.
    
    CV Section:
    {section}
    
    Job Description:
    {jd_text}
    
    Return only the rewritten section, nothing else.
    """
    try:
        config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            config=config,
            contents=prompt)
        return response.text
    except Exception as e:
        st.error(f"Generation error: {e}")
        return section

def generate_pdf(optimized_sections):
    """Generate PDF from optimized sections"""
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    for section in optimized_sections:
        # Handle both strings and tuples (original, optimized)
        if isinstance(section, tuple):
            _, optimized = section
        else:
            optimized = section
            
        pdf.multi_cell(0, 10, optimized)
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin-1')

if uploaded_file and jd_text:
    with st.spinner("Analyzing your CV..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            extracted_text = extract_text(tmp_path)
            cv_chunks = preprocess_text(extracted_text)
            jd_embedding = get_embedding(jd_text)
            
            if jd_embedding:
                match_score = calculate_match_score(cv_chunks, jd_embedding)
                display_match_score(match_score)
                
                relevant_sections = find_relevant_sections(cv_chunks, jd_embedding)
                st.session_state.optimized_sections = []
                
                if relevant_sections:
                    st.subheader("Optimization Suggestions")
                    for i, section in enumerate(relevant_sections, 1):
                        with st.expander(f"Section {i}", expanded=i==1):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Original**")
                                st.info(section)
                            with col2:
                                st.markdown("**Optimized**")
                                optimized = optimize_section(section, jd_text)
                                st.success(optimized)
                                st.session_state.optimized_sections.append((section, optimized))
        except Exception as e:
            st.error(f"Processing error: {e}")            

def generate_pdf(optimized_sections):
    """Generate PDF with robust Unicode handling"""
    pdf = PDF()
    pdf.add_page()
    
    # Font handling with better fallbacks
    try:
        # Try to use DejaVu font if available
        try:
            from fpdf import util
            font_path = util.get_fonts_dir() + '/DejaVuSans.ttf'
            pdf.add_font('DejaVu', '', font_path, uni=True)
            pdf.set_font('DejaVu', '', 11)
        except:
            # Try system-installed DejaVu
            pdf.add_font('DejaVu', '', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', uni=True)
            pdf.set_font('DejaVu', '', 11)
    except Exception as e:
        st.warning(f"Couldn't load DejaVu font: {str(e)}. Using Arial.")
        pdf.set_font("Arial", size=11)
    
    # Common Unicode character replacements
    char_replacements = {
        '\u2014': '--',  # Em dash
        '\u2013': '-',   # En dash
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u2022': '*',   # Bullet point
        '\u00b7': '*',   # Middle dot
    }
    
    for section in optimized_sections:
        if isinstance(section, tuple):
            _, optimized = section
        else:
            optimized = section
            
        # Ensure string type and replace special chars
        optimized = str(optimized)
        for char, replacement in char_replacements.items():
            optimized = optimized.replace(char, replacement)
            
        # Handle line breaks and encode safely
        optimized = optimized.encode('ascii', 'replace').decode('ascii')
        pdf.multi_cell(0, 10, optimized)
        pdf.ln(5)  # Add space between sections
    


