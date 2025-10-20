import os
import streamlit as st
from streamlit_lottie import st_lottie
# Streamlit page configuration must be the first Streamlit command
st.set_page_config(
    page_title="LeanAI Model MVP",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
from dotenv import load_dotenv
from openai import OpenAI
from mem0 import Memory
import supabase
from supabase.client import Client, ClientOptions
from pathlib import Path
import uuid
import sys
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import openai
import locale
from PyPDF2 import PdfReader, PdfWriter
from thefuzz import fuzz
import datetime
import pandas as pd
import json
import requests
import urllib.parse
import numpy as np
import io

# --- Custom CSS for beautiful UI ---
st.markdown("""
<style>
/* Global Styles */
body {
    background-color: #f5f7fa; /* Light background */
    font-family: 'Segoe UI', sans-serif;
}

/* Adjust main content area padding for spacious layout */
.main .block-container {
    padding-top: 40px;
    padding-right: 60px;
    padding-left: 60px;
    padding-bottom: 40px;
}

/* Sidebar styling */
.css-1d391kg, .css-1v0mbdj { /* Adjust these class names if needed based on Streamlit version */
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Slightly stronger shadow */
    margin-bottom: 20px;
}

/* Input fields */
input {
    border: 1px solid #e0e0e0; /* Lighter border */
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
    width: 100%;
    margin-bottom: 15px; /* Increased space */
    background-color: #f9f9f9; /* Slightly different input background */
}

/* Hide 'Press Enter to apply' text */
.stTextInput > div > div > input + div {
    display: none;
}

/* Buttons */
button[kind="primary"] {
    background-color: #1e73c4; /* A shade of blue */
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: bold;
    margin-top: 15px; /* Increased space */
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    transition: background-color 0.3s ease;
}

button[kind="primary"]:hover {
    background-color: #155b9e; /* Darker shade on hover */
}

/* Header styles */
h1, h2, h3, h4 {
    color: #333; /* Darker text for headers */
    font-weight: 700;
    margin-bottom: 15px;
    padding-top: 5px; /* Add some padding above headers */
}

/* Features layout (Cards) */
.feature-box {
    background-color: #ffffff;
    padding: 25px; /* Increased padding inside cards */
    border-radius: 16px; /* Rounded corners */
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1); /* More prominent shadow */
    margin-top: 15px; /* Increased margin */
    margin-bottom: 15px; /* Increased margin */
    text-align: left;
    height: 100%;
    transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth hover effect */
}

.feature-box:hover {
    transform: translateY(-5px); /* Lift effect on hover */
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15); /* Enhanced shadow on hover */
}

.stMarkdown > div > p {
    margin-bottom: 1rem; /* Standard space below paragraphs */
    color: #555; /* Slightly lighter text for paragraph */
}

/* Adjust spacing around columns */
.st-emotion-l8z9g2 > div { /* This targets the div inside the columns, adjust class name if needed */
    margin-bottom: 30px; /* Space between rows of columns */
    padding: 0 10px; /* Add horizontal padding between columns */
}

/* Center the main content block */
.css-18e3gdp.e8zbici2 {
    max-width: 1200px; /* Set a max width for content */
    margin: auto; /* Center the block */
}

/* Class for bold text */
.bold-text {
    font-weight: bold;
}

/* Dark mode adjustments */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #1e1e1e; /* Darker background */
    }
    .stApp {
        background-color: #1e1e1e;
    }
    .main .block-container {
        padding-top: 40px;
        padding-right: 60px;
        padding-left: 60px;
        padding-bottom: 40px;
    }
    .css-1d391kg, .css-1v0mbdj {
        background-color: #2c2c2c;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    input {
        border: 1px solid #444;
        background-color: #333;
        color: #f0f0f0;
    }
    button[kind="primary"] {
        background-color: #1a4d7d; /* Dark mode blue */
    }
    button[kind="primary"]:hover {
        background-color: #12395a; /* Darker shade on hover */
    }
    h1, h2, h3, h4 {
        color: #f0f0f0;
    }
    .feature-box {
        background-color: #2c2c2c;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.45);
    }
     .stMarkdown > div > p {
        color: #cccccc;
    }
}

</style>
""", unsafe_allow_html=True)

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path, override=True)

# Initialize Supabase client
supabase_url = os.environ.get("SUPABASE_URL", "")
supabase_key = os.environ.get("SUPABASE_KEY", "")
supabase_client = supabase.create_client(supabase_url, supabase_key)

model = os.getenv('MODEL_CHOICE', 'gpt-3.5-turbo')

# --- Gemini integration (minimal, no new requirements) ---
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'gemini').lower()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', None)
GEMINI_CHAT_MODEL = os.getenv('GEMINI_CHAT_MODEL', 'gemini-2.5-flash')
GEMINI_EMBED_MODEL = os.getenv('GEMINI_EMBED_MODEL', 'text-embedding-004')
GEMINI_CHAT_URL = f'https://generativelanguage.googleapis.com/v1/models/{GEMINI_CHAT_MODEL}:generateContent'
GEMINI_EMBED_URL = f'https://generativelanguage.googleapis.com/v1/models/{GEMINI_EMBED_MODEL}:embedContent'

# --- Configuration for LeanAI Model ---
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '200'))
SUBJECT_RAG_TOP_K = int(os.getenv('SUBJECT_RAG_TOP_K', '5'))
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1500'))  # tokens per chunk
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '200'))  # overlap between chunks

def gemini_chat(messages):
    """Call Gemini chat API with OpenAI-style messages."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    # Gemini expects a different message format
    # We'll concatenate all messages into a single prompt
    prompt = "\n".join([m['content'] for m in messages])
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_API_KEY}
    response = requests.post(GEMINI_CHAT_URL, params=params, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    candidates = response.json().get('candidates', [])
    if candidates:
        return candidates[0]['content']['parts'][0]['text']
    return "[No response from Gemini]"

def gemini_embed(text):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    try:
        payload = {
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }
        headers = {"Content-Type": "application/json"}
        params = {"key": GEMINI_API_KEY}
        response = requests.post(
            GEMINI_EMBED_URL, 
            params=params, 
            headers=headers, 
            data=json.dumps(payload),
            timeout=30
        )
        if response.status_code != 200:
            st.error(f"Gemini API Error: {response.status_code} - {response.text}")
            response.raise_for_status()
        response_data = response.json()
        if 'embedding' in response_data:
            embedding = response_data['embedding'].get('values', None)
        elif 'data' in response_data:
            embedding = response_data['data'][0].get('embedding', None)
        else:
            st.error(f"Unexpected response format: {response_data}")
            raise ValueError("No embedding found in response")
        if not embedding:
            raise ValueError("No embedding returned from Gemini API")
        return embedding
    except requests.exceptions.ConnectionError as e:
        st.error(f"Connection error to Gemini API: {str(e)}")
        st.warning("Please check your internet connection and Gemini API key.")
        raise
    except requests.exceptions.Timeout as e:
        st.error(f"Timeout error to Gemini API: {str(e)}")
        st.warning("The request to Gemini API timed out. Please try again.")
        raise
    except requests.exceptions.RequestException as e:
        st.error(f"Request error to Gemini API: {str(e)}")
        st.warning("There was an error communicating with Gemini API.")
        raise
    except Exception as e:
        st.error(f"Unexpected error in gemini_embed: {str(e)}")
        raise

# Cache OpenAI client and Memory instance
@st.cache_resource
def get_openai_client():
    return OpenAI()

@st.cache_resource
def get_memory():
    class NoopMemory:
        def search(self, query: str = "", user_id: str = "", limit: int = 3):
            return {"results": []}

        def add(self, *args, **kwargs):
            return None

    # Validate connection string first
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        st.warning("DATABASE_URL not set. Memory features are disabled.")
        return NoopMemory()

    config = {
        "llm": {
            "provider": LLM_PROVIDER,
            "config": {
                "model": model
            }
        },
        "vector_store": {
            "provider": "supabase",
            "config": {
                "connection_string": db_url,
                "collection_name": "memories"
            }
        }
    }

    try:
        return Memory.from_config(config)
    except Exception as e:
        st.error(f"Memory backend init failed; running without vector memory. Error: {str(e)}")
        return NoopMemory()

# Get cached resources
openai_client = get_openai_client()
memory = get_memory()

# --- Document Chunking Utility ---
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Split text into chunks with overlap for better context preservation.
    Simple sentence-based chunking with character limits.
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size, start a new chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from the end of previous chunk
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += (". " if current_chunk else "") + sentence
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_file_content(file):
    """Extract content from different file types"""
    try:
        file_type = file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            # Handle PDF files
            pdf_reader = PdfReader(file)
            content = ""
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
            return content
            
        elif file_type == 'txt':
            # Handle text files
            return file.getvalue().decode("utf-8")
            
        elif file_type == 'docx':
            # Handle Word documents
            import docx  
            doc = docx.Document(file)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        raise Exception(f"Error extracting file content: {str(e)}")

def ensure_vector(embedding):
    # If it's a string, try to parse as JSON
    if isinstance(embedding, str):
        try:
            embedding = json.loads(embedding)
        except Exception:
            raise ValueError("Embedding string could not be parsed as JSON array.")
    # If it's a list of numbers (possibly as strings), convert all to float
    if isinstance(embedding, list):
        # If it's a list of lists (shouldn't happen for a single embedding), flatten
        if len(embedding) > 0 and isinstance(embedding[0], list):
            embedding = embedding[0]
        return [float(x) for x in embedding]
    # If it's a single float or int, raise error
    if isinstance(embedding, (float, int)):
        raise ValueError("Embedding is a single number, expected a list of floats.")
    raise ValueError(f"Unexpected embedding type: {type(embedding)}")

def ensure_vector_array_for_pgvector_array(embeddings):
    """Ensure value is a list of vectors (list[list[float]]) for pgvector[] column."""
    normalized_list = []
    # If a single vector was provided, wrap it
    if isinstance(embeddings, (list, tuple)) and (len(embeddings) == 0 or isinstance(embeddings[0], (int, float, str, list, tuple))):
        # Detect single vector (list of numbers)
        if len(embeddings) == 0 or isinstance(embeddings[0], (int, float, str)):
            embeddings = [embeddings]
    for vec in embeddings:
        v = ensure_vector(vec)
        # Ensure final is list[float]
        v = [float(x) for x in v]
        normalized_list.append(v)
    return normalized_list

# --- Subject Management Functions (transformed from customer functions) ---

def generate_subject_id():
    """Generate a unique subject ID"""
    return str(uuid.uuid4())

def generate_subject_display_id():
    """Generate a human-readable subject ID in format LC-YYYY-SUBJ-XXXX"""
    year = datetime.datetime.now().year
    
    # Get all subject IDs to find the highest number
    response = supabase_client.table('subjects').select('display_id').execute()
    max_num = 0
    
    if response.data:
        for subject in response.data:
            display_id = subject.get('display_id', '')
            # Check if the ID matches our format (LC-YYYY-SUBJ-XXXX)
            if isinstance(display_id, str) and display_id.startswith(f'LC-{year}-SUBJ-'):
                try:
                    num = int(display_id.split('-')[-1])
                    max_num = max(max_num, num)
                except ValueError:
                    continue
    
    # Increment the highest number found
    new_num = max_num + 1
    return f"LC-{year}-SUBJ-{new_num:04d}"

def find_similar_subjects(subject_name: str, threshold: int = 80):
    """Find similar subject names using fuzzy matching"""
    response = supabase_client.table('subjects').select('subject_name,subject_id').execute()
    similar_subjects = []
    
    for subject in response.data:
        # Calculate similarity score
        ratio = fuzz.ratio(subject_name.lower(), subject['subject_name'].lower())
        if ratio >= threshold:
            similar_subjects.append({
                'name': subject['subject_name'],
                'id': subject['subject_id'],
                'similarity': ratio
            })
    
    return sorted(similar_subjects, key=lambda x: x['similarity'], reverse=True)

def generate_subject_profile(subject_name: str, user_id: str):
    """Generate a subject matter profile using AI"""
    
    # Create the enhanced prompt for profile generation
    system_prompt = f"""You are an AI Subject Matter Analyst for LeanAI Model MVP. Your mission is to create a comprehensive profile for the subject matter: "{subject_name}".

Your task is to analyze and structure information about this subject matter to create a detailed knowledge base entry.

🧾 Primary Deliverables

Subject Overview:
- Provide a concise summary (≤300 words) of the subject matter's core concepts, scope, and relevance
- Identify key domains, applications, and stakeholders involved
- Highlight current trends, challenges, and opportunities

Knowledge Structure:
Create a structured breakdown including:
- Core Concepts: Fundamental principles and definitions
- Key Components: Main elements, processes, or methodologies
- Applications: Real-world use cases and implementations  
- Relationships: How this subject connects to other domains
- Current State: Recent developments and market status

Research Context:
- Important questions that need investigation
- Knowledge gaps and areas for further research
- Potential sources of information and expertise
- Methodologies for deeper analysis

Strategic Insights:
- Why this subject matter is important
- Potential business or research opportunities
- Risk factors and challenges to consider
- Future outlook and emerging trends

Return a comprehensive profile that will serve as the foundation for building knowledge about "{subject_name}".
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate a comprehensive profile for the subject matter: {subject_name}"}
    ]
    
    # Get response and convert to string
    response = gemini_chat(messages)
    profile_text = response
    return profile_text

def create_new_subject(subject_name: str, user_id: str):
    """Handle the complete subject creation workflow"""
    # Ensure we have a valid state
    if 'subject_creation_state' not in st.session_state or st.session_state.subject_creation_state is None:
        st.session_state.subject_creation_state = {
            'step': 1,
            'subject_name': subject_name,
            'profile': None,
            'confirmed': False
        }

    state = st.session_state.subject_creation_state
    
    # Step 1: Check for similar subjects
    if state['step'] == 1:
        similar_subjects = find_similar_subjects(subject_name)
        
        if similar_subjects:
            st.warning(f"Similar subject found: {similar_subjects[0]['name']}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Create New Subject Anyway"):
                    state['step'] = 2
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.subject_creation_state = None
                    st.rerun()
            return None
        else:
            state['step'] = 2
            st.rerun()
    
    # Step 2: Generate subject profile
    if state['step'] == 2:
        with st.spinner("Generating subject matter profile..."):
            profile = generate_subject_profile(subject_name, user_id)
            state['profile'] = profile
            st.write("Generated Profile:")
            st.write(profile)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm and Add to Knowledge Base"):
                    state['confirmed'] = True
                    state['step'] = 3
                    st.rerun()
            with col2:
                if st.button("Cancel"):
                    st.session_state.subject_creation_state = None
                    st.rerun()
            return None
    
    # Step 3: Create database entry
    if state['step'] == 3 and state['confirmed']:
        subject_id = generate_subject_id()
        display_id = generate_subject_display_id()
        profile_input = f"Create profile for {subject_name}"
        profile_output = str(state['profile'])
        
        # Generate embedding for the profile input
        embedding = gemini_embed(profile_input)
        interaction_embeddings = ensure_vector_array_for_pgvector_array(ensure_vector(embedding))
        
        interaction_json = {
            "input": profile_input,
            "output": profile_output,
            "timestamp": datetime.datetime.now().isoformat(),
            "user_id": user_id
        }
        
        data = {
            "subject_id": subject_id,
            "display_id": display_id,
            "subject_name": subject_name,
            "input_conversation": [profile_input],
            "output_conversation": [profile_output],
            "interaction_metadata": [interaction_json],  # list of dicts (JSON)
            "interaction_embeddings": interaction_embeddings,  # list of lists of floats (vector per interaction)
            "created_by": user_id
        }
        
        try:
            response = supabase_client.table('subjects').insert(data).execute()
            if response.data:
                # Clear the creation state first
                st.session_state.subject_creation_state = None
                return response.data[0]
            else:
                st.error("Failed to create subject")
                return None
        except Exception as e:
            st.error(f"Error creating subject: {str(e)}")
            return None

# Authentication functions (reused from CRM)
def sign_up(email, password, full_name):
    try:
        response = supabase_client.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "full_name": full_name
                }
            }
        })
        if response and response.user:
            st.success("Sign up successful! Please check your email and confirm your address before logging in.")
        return response
    except Exception as e:
        st.error(f"Error signing up: {str(e)}")
        return None

def sign_in(email, password):
    try:
        response = supabase_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if response and response.user:
            # Store user info directly in session state
            st.session_state.authenticated = True
            st.session_state.user = response.user
            st.rerun()
        return response
    except Exception as e:
        st.error(f"Error signing in: {str(e)}")
        return None

def sign_out():
    try:
        supabase_client.auth.sign_out()
        # Clear only authentication-related session state
        st.session_state.authenticated = False
        st.session_state.user = None
        # Set a flag to trigger rerun on next render
        st.session_state.logout_requested = True
    except Exception as e:
        st.error(f"Error signing out: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_all_subject_names():
    response = supabase_client.table('subjects').select('subject_name,subject_id').execute()
    if response.data:
        return {s['subject_name']: s['subject_id'] for s in response.data}
    return {}

def get_subject_interactions(subject_id: str):
    """Fetch all interactions for a specific subject from the subjects table"""
    try:
        response = supabase_client.table('subjects').select('*').eq('subject_id', subject_id).single().execute()
        if response.data:
            # Combine input and output conversations into interactions
            interactions = []
            for i, (input_msg, output_msg) in enumerate(zip(
                response.data.get('input_conversation', []),
                response.data.get('output_conversation', [])
            )):
                interactions.append({
                    'interaction_input': input_msg,
                    'llm_output_summary': output_msg,
                    'created_at': response.data.get('created_at')  # Using the subject's creation time
                })
            return interactions
        return []
    except Exception as e:
        st.error(f"Error fetching interactions: {str(e)}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def update_subject_interaction(subject_id: str, new_input: str, new_output: str, user_id: str):
    """Update subject interaction, storing a structured JSON object in the metadata."""
    # 1. Generate embedding for the new input
    embedding = gemini_embed(new_input)
    import numpy as np
    # Normalize embedding to list[float]
    embedding = ensure_vector(embedding)
    
    # 2. Create the new interaction object (as JSON)
    new_interaction_json = {
        "input": new_input,
        "output": new_output,
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id
    }

    # 3. Fetch existing data
    subject = supabase_client.table('subjects').select("*").eq('subject_id', subject_id).single().execute()
    inps = subject.data.get('input_conversation') or []
    outs = subject.data.get('output_conversation') or []
    embs = subject.data.get('interaction_embeddings') or []
    metas = subject.data.get('interaction_metadata') or []

    # Defensive: ensure all are lists
    if not isinstance(inps, list):
        inps = list(inps) if inps else []
    if not isinstance(outs, list):
        outs = list(outs) if outs else []
    if not isinstance(embs, list):
        embs = list(embs) if embs else []
    if not isinstance(metas, list):
        metas = list(metas) if metas else []

    # 4. Append new data
    updated_inputs = inps + [new_input]
    updated_outputs = outs + [new_output]
    updated_embs = embs + [embedding]
    updated_metas = metas + [new_interaction_json]

    # Normalize to list of float vectors for pgvector[]
    updated_embs = ensure_vector_array_for_pgvector_array(updated_embs)
    # Ensure every metadata is a dict
    updated_metas = [m if isinstance(m, dict) else json.loads(m) for m in updated_metas]

    # 5. Save
    try:
        response = supabase_client.table('subjects').update({
            'input_conversation': updated_inputs,
            'output_conversation': updated_outputs,
            'interaction_embeddings': updated_embs,
            'interaction_metadata': updated_metas,
            'updated_at': datetime.datetime.now().isoformat()
        }).eq('subject_id', subject_id).execute()
        
        return response.data
    except Exception as e:
        st.error(f"Supabase update error: {e}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def retrieve_relevant_subject_interactions(subject_id: str, query: str, top_k: int = 3):
    """Retrieve the most relevant past interactions using vector similarity, returning full JSON objects."""
    query_embedding = gemini_embed(query)
    import json
    import numpy as np
    
    if isinstance(query_embedding, str):
        try:
            query_embedding = json.loads(query_embedding)
        except Exception:
            query_embedding = [float(query_embedding)]
    elif isinstance(query_embedding, float) or isinstance(query_embedding, int):
        query_embedding = [query_embedding]
    elif isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    
    subject = supabase_client.table('subjects').select("interaction_embeddings, interaction_metadata").eq('subject_id', subject_id).single().execute()
    
    if not subject.data:
        return []

    embs = subject.data.get('interaction_embeddings', [])
    metas = subject.data.get('interaction_metadata', [])

    if not embs or not metas:
        return []

    # Handle case where embeddings list is shorter than metadata list, or vice-versa
    min_len = min(len(embs), len(metas))
    embs_np = np.array(embs[:min_len])
    metas = metas[:min_len]

    if embs_np.shape[0] == 0:
        return []

    query_np = np.array(query_embedding)
    similarities = embs_np @ query_np / (np.linalg.norm(embs_np, axis=1) * np.linalg.norm(query_np) + 1e-8)
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return the full JSON object from metadata, adding similarity score
    results = []
    for i in top_indices:
        interaction_json = metas[i]
        interaction_json['similarity'] = float(similarities[i])
        results.append(interaction_json)
        
    return results

def process_uploaded_file_with_chunking(file, subject_id: str, user_id: str):
    """Process uploaded file, extract content, chunk it, and store chunks with embeddings."""
    try:
        # Extract file content based on file type
        file_content = extract_file_content(file)
        
        # Chunk the content for large files
        chunks = chunk_text(file_content)
        
        # Create a summary of the file content
        summary_prompt = f"""Analyze the following document content for knowledge base purposes:
{file_content[:3000]}...  # Truncate for prompt efficiency

Format the analysis to include:
1. Document Overview: Main topic and purpose
2. Key Concepts: Important ideas and definitions
3. Structure: How information is organized
4. Insights: Notable findings or conclusions
5. Knowledge Value: How this contributes to understanding the subject matter

Provide a comprehensive but concise analysis suitable for a knowledge management system.
"""

        messages = [
            {"role": "system", "content": "You are a document analysis assistant specialized in knowledge extraction and summarization. Analyze the content for knowledge base purposes, focusing on key concepts and insights."},
            {"role": "user", "content": summary_prompt}
        ]

        # Get AI summary
        summary = gemini_chat(messages)
        
        # Store chunks in subject_documents table
        stored_chunks = []
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding for each chunk
                chunk_embedding = gemini_embed(chunk)
                chunk_embedding = ensure_vector(chunk_embedding)
                
                # Store chunk in subject_documents table
                chunk_data = {
                    "subject_id": subject_id,
                    "content": chunk,
                    "embedding": chunk_embedding,
                    "metadata": {
                        "filename": file.name,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "user_id": user_id,
                        "source": "file_upload"
                    }
                }
                
                response = supabase_client.table('subject_documents').insert(chunk_data).execute()
                if response.data:
                    stored_chunks.append(response.data[0])
                    
            except Exception as e:
                st.warning(f"Failed to store chunk {i}: {str(e)}")
        
        return True, f"Document processed successfully! Created {len(stored_chunks)} chunks. Review analysis below and save.", file_content, summary
        
    except Exception as e:
        return False, f"Error processing file: {str(e)}", None, None

def delete_subject_interaction(subject_id: str, interaction_index: int):
    """Delete a single interaction (and aligned embedding/metadata) by index for a subject."""
    try:
        subject = supabase_client.table('subjects').select("*").eq('subject_id', subject_id).single().execute()
        if not subject.data:
            st.error("Subject not found.")
            return False

        input_conversation = subject.data.get('input_conversation') or []
        output_conversation = subject.data.get('output_conversation') or []
        interaction_embeddings = subject.data.get('interaction_embeddings') or []
        interaction_metadata = subject.data.get('interaction_metadata') or []

        # Validate index against the shortest list among the aligned arrays
        aligned_len = min(len(input_conversation), len(output_conversation), len(interaction_embeddings), len(interaction_metadata))
        if interaction_index < 0 or interaction_index >= aligned_len:
            st.error("Invalid interaction index.")
            return False

        def remove_at(lst):
            if isinstance(lst, list) and 0 <= interaction_index < len(lst):
                return lst[:interaction_index] + lst[interaction_index+1:]
            return lst

        updated_inputs = remove_at(input_conversation)
        updated_outputs = remove_at(output_conversation)
        updated_embs = remove_at(interaction_embeddings)
        updated_metas = remove_at(interaction_metadata)

        response = supabase_client.table('subjects').update({
            'input_conversation': updated_inputs,
            'output_conversation': updated_outputs,
            'interaction_embeddings': updated_embs,
            'interaction_metadata': updated_metas,
            'updated_at': datetime.datetime.now().isoformat()
        }).eq('subject_id', subject_id).execute()

        return bool(response.data)
    except Exception as e:
        st.error(f"Error deleting interaction: {str(e)}")
        return False

def delete_subject(subject_id: str):
    """Delete a subject and all their data from the subjects table."""
    try:
        # First delete all associated documents
        supabase_client.table('subject_documents').delete().eq('subject_id', subject_id).execute()
        # Then delete the subject
        response = supabase_client.table('subjects').delete().eq('subject_id', subject_id).execute()
        return bool(response.data)
    except Exception as e:
        st.error(f"Error deleting subject: {str(e)}")
        return False

# Add a function to load Lottie animation JSON from a URL.
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        # Fail-safe: if network/DNS fails, skip animation
        return None

# --- UI Rendering Functions ---

def render_subject_creation_ui_tab(user_id):
    st.subheader("Create New Subject Matter")

    # Initialize the input field state if it doesn't exist
    if 'tab_new_subject_name' not in st.session_state:
        st.session_state.tab_new_subject_name = ""

    # Text input for subject name
    new_subject_name = st.text_input("Enter Subject Matter Name", key="tab_new_subject_name")

    # Add the Enter button always below the input field
    if st.button("Enter", key="create_subject_enter_button", type="primary"):
        if not new_subject_name:
            st.warning("Please enter a subject matter name.")
        else:
            # Initialize the creation state
            st.session_state.subject_creation_state = {
                'step': 1,
                'subject_name': new_subject_name,
                'profile': None,
                'confirmed': False
            }
            st.rerun()

    # --- Subject creation flow logic based on state ---
    if st.session_state.get('subject_creation_state') is not None:
        # Call create_new_subject to handle the current step (1, 2, or 3)
        created_subject_data = create_new_subject(
            st.session_state.subject_creation_state['subject_name'],
            user_id
        )

        # After create_new_subject finishes its steps, check if creation was successful (step 3 completed)
        if created_subject_data:
            # Subject creation finished successfully in create_new_subject (step 3)
            # Clear the creation state
            st.session_state.subject_creation_state = None
            # Show success message
            st.success(f"Subject Matter '{created_subject_data.get('subject_name', 'created')}' created successfully! (ID: {created_subject_data.get('display_id', 'N/A')})")
            st.info("To update your knowledge about this subject, please go to 'Manage MVP' section, select the subject, and upload your documents and insights.")

def render_manage_subject_ui(user_id: str):
    """Render the Manage MVP window UI."""

    # State 1: Subject Selection
    if 'selected_subject_for_update' not in st.session_state or st.session_state['selected_subject_for_update'] is None:
        st.subheader("Select Subject Matter to Manage")

        subjects_dict = get_all_subject_names()
        if not subjects_dict:
            st.warning("No subjects found. Please create a subject first.")
            return

        sorted_subject_names = sorted(subjects_dict.keys())

        col_dropdown, col_button = st.columns([0.7, 0.3])

        with col_dropdown:
            selected_subject_name = st.selectbox(
                "Select Subject Matter",
                options=sorted_subject_names,
                key="update_interaction_subject_select"
            )

        with col_button:
            st.markdown("<br>", unsafe_allow_html=True) # Add a small space above the button
            select_button = st.button("Select", key="select_subject_button", type="primary")

        # Store the selected subject ID if the button is clicked and trigger rerun
        if select_button and selected_subject_name:
            selected_subject_id = subjects_dict.get(selected_subject_name)
            if selected_subject_id:
                st.session_state['selected_subject_for_update'] = {
                    'name': selected_subject_name,
                    'id': selected_subject_id
                }
                st.rerun() # Rerun to show the interaction details

    # State 2, 3, 4: Interaction Details and Adding New Data
    else:
        selected_subject = st.session_state['selected_subject_for_update']
        subject_name = selected_subject['name']
        subject_id = selected_subject['id']

        st.subheader(f"Knowledge Base: {subject_name}")

        # Display interaction history (Collapsible Cards)
        interactions = get_subject_interactions(subject_id)
        if interactions:
            for idx, interaction in enumerate(reversed(interactions)):
                with st.expander(f"{interaction['created_at']} | Entry #{len(interactions)-idx}"):
                    st.markdown(f"**Input:** {interaction['interaction_input']}")
                    st.markdown(f"**AI Analysis:** {interaction['llm_output_summary']}")
                    # Delete button for this interaction
                    col_a, col_b = st.columns([0.2, 0.8])
                    with col_a:
                        # Map reversed display index back to original index in arrays
                        original_index = len(interactions) - 1 - idx
                        if st.button("🗑️ Delete", key=f"delete_interaction_{original_index}"):
                            success = delete_subject_interaction(subject_id, original_index)
                            if success:
                                st.success("Entry deleted.")
                                st.rerun()
                            else:
                                st.error("Failed to delete entry.")
            if st.button("Select Another Subject"):
                st.session_state['selected_subject_for_update'] = None
                if 'current_interaction_analysis' in st.session_state:
                    st.session_state['current_interaction_analysis'] = None
                if 'current_file_analysis' in st.session_state:
                    st.session_state['current_file_analysis'] = None
                st.rerun()
        else:
            st.info("No knowledge entries found for this subject yet.")
            if st.button("Select Another Subject"):
                st.session_state['selected_subject_for_update'] = None
                if 'current_interaction_analysis' in st.session_state:
                    st.session_state['current_interaction_analysis'] = None
                if 'current_file_analysis' in st.session_state:
                    st.session_state['current_file_analysis'] = None
                st.rerun()

        st.markdown("\n---\n")

        # --- Section for adding New Knowledge and Uploads ---
        st.subheader("Add New Knowledge or Upload Document")

        # State 3: Display file analysis and Save/Cancel buttons if available in session state
        if 'current_file_analysis' in st.session_state and st.session_state['current_file_analysis'] is not None:
            file_analysis_data = st.session_state['current_file_analysis']
            st.subheader("📄 File Analysis Ready to Save")
            st.write(f"**File:** {file_analysis_data['file_name']}")
            st.markdown("**Extracted Content (Input):**")
            st.expander("View full content").markdown(f"```\n{file_analysis_data['file_content'][:1000]}...\n```")
            st.markdown("**AI Summary (Output):**")
            st.write(file_analysis_data['summary'])
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save File Analysis", key="save_file_interaction_button"):
                    llm_output = f"File Analysis:\n{file_analysis_data['summary']}"
                    interaction_input = f"Uploaded file: {file_analysis_data['file_name']}. Content summary: {file_analysis_data['file_content'][:200]}..."
                    result = update_subject_interaction(subject_id, interaction_input, llm_output, user_id)
                    if result:
                        st.success(f"File analysis for {file_analysis_data['file_name']} saved successfully!")
                        st.session_state['current_file_analysis'] = None
                        st.rerun()
                    else:
                        st.error("Failed to save file analysis")
            with col2:
                if st.button("❌ Cancel", key="cancel_file_interaction_button"):
                    st.session_state['current_file_analysis'] = None
                    st.rerun()
            st.markdown("\n---\n")

        # State 4: Display analysis and Save/Cancel buttons for text interaction if available in session state
        elif 'current_interaction_analysis' in st.session_state and st.session_state['current_interaction_analysis'] is not None:
            analysis_data = st.session_state['current_interaction_analysis']
            st.subheader("🤖 AI Analysis")
            st.write(analysis_data['analysis'])
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Knowledge Entry", key="save_interaction_button"):
                    result = update_subject_interaction(subject_id, analysis_data['new_interaction'], analysis_data['analysis'], user_id)
                    if result:
                        st.success("Knowledge entry saved successfully!")
                        st.session_state['current_interaction_analysis'] = None
                        st.rerun()
                    else:
                        st.error("Failed to save knowledge entry")
            with col2:
                if st.button("❌ Cancel", key="cancel_interaction_button"):
                    st.session_state['current_interaction_analysis'] = None
                    st.rerun()
            st.markdown("\n---\n")

        # State 5: Default state for adding new knowledge/uploads
        else:
            st.subheader("✍️ New Knowledge Entry")
            st.caption("Add new insights, observations, or questions about this subject matter.")
            new_interaction = st.text_area("📝 Enter new knowledge, insights, or questions", key="new_interaction_textarea")
            if st.button("💡 Analyze with AI", key="analyze_interaction_button") and new_interaction:
                with st.spinner("Analyzing knowledge entry..."):
                    # Create analysis prompt for subject matter
                    system_prompt = f"""You are an AI Knowledge Analyst for the subject matter: "{subject_name}".

Analyze the following input and provide:

1. **Key Insights**: What are the main points or discoveries?
2. **Knowledge Integration**: How does this relate to existing knowledge about {subject_name}?
3. **Implications**: What are the broader implications or applications?
4. **Questions Raised**: What new questions or areas for investigation does this suggest?
5. **Action Items**: What should be done next to build on this knowledge?

Provide a structured, comprehensive analysis that adds value to the knowledge base."""

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": new_interaction}
                    ]
                    
                    analysis = gemini_chat(messages)
                    st.session_state['current_interaction_analysis'] = {
                        'new_interaction': new_interaction,
                        'analysis': analysis
                    }
                st.rerun()
            st.markdown("\n---\n")
            st.subheader("Upload Document")
            uploaded_file = st.file_uploader(
                "Upload a document (PDF, TXT, DOCX)",
                type=['pdf', 'txt', 'docx'],
                key="subject_file_upload"
            )
            if uploaded_file:
                if st.button("Process Document", key="process_uploaded_file_button"):
                    success, message, file_content, summary = process_uploaded_file_with_chunking(uploaded_file, subject_id, user_id)
                    if success:
                        st.success(message)
                        st.session_state['current_file_analysis'] = {
                            'file_name': uploaded_file.name,
                            'file_content': file_content,
                            'summary': summary
                        }
                        st.rerun()
                    else:
                        st.error(message)

        # --- Danger Zone: Delete Subject ---
        st.markdown("\n---\n")
        st.subheader("⚠️ Danger Zone")
        st.caption("Deleting a subject will remove all knowledge entries and documents and cannot be undone.")
        confirm = st.checkbox("I understand this action is irreversible.", key="confirm_delete_subject")
        if st.button("🗑️ Delete Subject", key="delete_subject_button"):
            if confirm:
                with st.spinner("Deleting subject..."):
                    ok = delete_subject(subject_id)
                if ok:
                    st.success(f"Subject '{subject_name}' deleted.")
                    # Clear selection and any transient state
                    st.session_state['selected_subject_for_update'] = None
                    if 'current_interaction_analysis' in st.session_state:
                        st.session_state['current_interaction_analysis'] = None
                    if 'current_file_analysis' in st.session_state:
                        st.session_state['current_file_analysis'] = None
                    st.rerun()
                else:
                    st.error("Failed to delete subject. Please try again.")
            else:
                st.warning("Please confirm the irreversible action before deleting.")

def render_subject_rag_ui(user_id: str):
    """Render the RAG for MVP window UI - One-page narration with document grounding."""
    st.title("📖 RAG for MVP: One-Page Narration")
    st.write("Generate comprehensive narratives grounded in your subject matter knowledge base.")
    st.markdown("---")

    # Subject Selection
    subjects_dict = get_all_subject_names()
    if not subjects_dict:
        st.warning("No subjects found. Please create a subject first.")
        return

    sorted_subject_names = sorted(subjects_dict.keys())
    selected_subject_name = st.selectbox(
        "Select Subject Matter",
        options=sorted_subject_names,
        key="rag_subject_select"
    )
    subject_id = subjects_dict[selected_subject_name]

    st.markdown("---")

    # Optional new document upload
    st.subheader("📤 Optional: Upload New Document")
    st.caption("Upload a document to include in the narrative generation (will be processed and stored)")
    
    uploaded_file = st.file_uploader(
        "Upload document for analysis (PDF, TXT, DOCX)",
        type=['pdf', 'txt', 'docx'],
        key="rag_file_upload"
    )
    
    new_document_content = ""
    if uploaded_file:
        if st.button("Process New Document", key="process_rag_document"):
            with st.spinner("Processing document..."):
                success, message, file_content, summary = process_uploaded_file_with_chunking(uploaded_file, subject_id, user_id)
                if success:
                    st.success(f"Document processed and added to knowledge base!")
                    new_document_content = file_content
                    st.info("Document content will be included in the narrative generation.")
                else:
                    st.error(message)

    st.markdown("---")

    # Narrative Generation
    st.subheader("📝 Generate One-Page Narrative")
    st.caption("Create a comprehensive narrative integrating all knowledge about the selected subject matter")
    
    narrative_focus = st.text_area(
        "Narrative Focus (optional)",
        placeholder="Specify particular aspects you want the narrative to focus on, or leave blank for comprehensive overview",
        key="narrative_focus"
    )
    
    if st.button("🎯 Generate Narrative", key="generate_narrative_button", type="primary"):
        with st.spinner("Generating comprehensive narrative..."):
            # Retrieve relevant interactions
            query = f"comprehensive overview of {selected_subject_name}" + (f" focusing on {narrative_focus}" if narrative_focus else "")
            relevant_interactions = retrieve_relevant_subject_interactions(subject_id, query, top_k=SUBJECT_RAG_TOP_K)
            
            # Retrieve relevant document chunks
            try:
                query_embedding = gemini_embed(query)
                query_embedding = ensure_vector(query_embedding)
                
                # Get relevant document chunks
                doc_response = supabase_client.rpc(
                    'match_subject_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_count': SUBJECT_RAG_TOP_K,
                        'match_threshold': 0.3,
                        'subject_filter': subject_id
                    }
                ).execute()
                
                relevant_documents = doc_response.data if doc_response.data else []
            except Exception as e:
                st.warning(f"Document retrieval error: {str(e)}")
                relevant_documents = []
            
            # Build context from retrieved information
            context = f"Subject Matter: {selected_subject_name}\n\n"
            
            if relevant_interactions:
                context += "=== KNOWLEDGE BASE ENTRIES ===\n"
                for i, interaction in enumerate(relevant_interactions, 1):
                    context += f"\nEntry {i} (Relevance: {interaction.get('similarity', 0):.2f}):\n"
                    context += f"Input: {interaction.get('input', '')}\n"
                    context += f"Analysis: {interaction.get('output', '')}\n"
                context += "\n"
            
            if relevant_documents:
                context += "=== DOCUMENT EXCERPTS ===\n"
                for i, doc in enumerate(relevant_documents, 1):
                    context += f"\nDocument {i} (Relevance: {doc.get('similarity', 0):.2f}):\n"
                    context += f"Source: {doc.get('metadata', {}).get('filename', 'Unknown')}\n"
                    context += f"Content: {doc.get('content', '')[:500]}...\n"
                context += "\n"
            
            if new_document_content:
                context += "=== NEWLY UPLOADED DOCUMENT ===\n"
                context += f"Content: {new_document_content[:1000]}...\n\n"
            
            # One-page narration system prompt
            system_prompt = """You are LeanAI MVP Narrator. Using the retrieved context and any newly uploaded document(s), produce a single page narrative with the following structure:

1. **Purpose and Scope**: What this subject matter encompasses and why it matters
2. **Key Facts and Constraints**: Essential information, limitations, and boundaries
3. **Current Understanding**: What we know based on the knowledge base
4. **Open Questions and Assumptions**: What remains unclear or assumed
5. **Next Steps**: Recommended actions for further development

Guidelines:
- Keep to 400-600 words total
- Be precise and structured
- Avoid fluff or repetition  
- Integrate information from all sources coherently
- Highlight gaps in knowledge
- Focus on actionable insights

If the context is insufficient, clearly state what information is missing."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a one-page narrative for: {selected_subject_name}\n\nFocus: {narrative_focus if narrative_focus else 'Comprehensive overview'}\n\nContext:\n{context}"}
            ]
            
            narrative = gemini_chat(messages)
            
            # Display the generated narrative
            st.subheader("📖 Generated Narrative")
            st.markdown(narrative)
            
            # Option to save the narrative
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Narrative to Knowledge Base", key="save_narrative_button"):
                    narrative_input = f"Generated comprehensive narrative for {selected_subject_name}" + (f" focusing on: {narrative_focus}" if narrative_focus else "")
                    result = update_subject_interaction(subject_id, narrative_input, narrative, user_id)
                    if result:
                        st.success("Narrative saved to knowledge base!")
                    else:
                        st.error("Failed to save narrative")
            
            with col2:
                # Download as text file
                st.download_button(
                    label="📥 Download Narrative",
                    data=narrative,
                    file_name=f"{selected_subject_name.replace(' ', '_')}_narrative.txt",
                    mime="text/plain"
                )
            
            # Show what sources were used
            if relevant_interactions or relevant_documents:
                with st.expander("📚 Sources Used in Narrative"):
                    if relevant_interactions:
                        st.write("**Knowledge Base Entries:**")
                        for i, interaction in enumerate(relevant_interactions, 1):
                            st.write(f"{i}. Relevance: {interaction.get('similarity', 0):.2f} - {interaction.get('input', '')[:100]}...")
                    
                    if relevant_documents:
                        st.write("**Document Sources:**")
                        for i, doc in enumerate(relevant_documents, 1):
                            filename = doc.get('metadata', {}).get('filename', 'Unknown')
                            st.write(f"{i}. {filename} (Relevance: {doc.get('similarity', 0):.2f})")

def render_apply_to_leanchems_ui(user_id: str):
    """Module 4: Apply selected subject to LeanChems situation and save application record."""
    st.title("📌 Apply to LeanChems")
    subjects_dict = get_all_subject_names()
    if not subjects_dict:
        st.warning("No subjects found. Please create a subject first.")
        return
    sorted_subject_names = sorted(subjects_dict.keys())
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        selected_subject_name = st.selectbox("Select Subject", options=sorted_subject_names, key="apply_subject_select")
        narrative = st.text_area("Describe LeanChems situation", key="apply_narrative", height=180)
        uploaded_file = st.file_uploader("Optional: Upload supporting file (PDF, TXT, DOCX)", type=["pdf", "txt", "docx"], key="apply_upload")
        run = st.button("Run Subject Mapping", key="apply_run", type="primary")
    with col2:
        if run and selected_subject_name and narrative:
            with st.spinner("Analyzing and mapping to framework..."):
                subject_id = subjects_dict[selected_subject_name]

                # Read optional uploaded file content
                file_text = ""
                if uploaded_file is not None:
                    try:
                        file_text = extract_file_content(uploaded_file)
                    except Exception as e:
                        st.warning(f"Could not read uploaded file: {e}")

                # LLM-driven parsing to structured layers
                system_prompt = (
                    "You are an information extraction system that structures a business narrative into layers: "
                    "Context, Drivers, Metrics, Scenarios. Return strict JSON only."
                )
                user_prompt = (
                    f"Subject: {selected_subject_name}\n\n"
                    f"LeanChems Narrative:\n{narrative}\n\n"
                    f"Supporting Document (optional, may be empty):\n{file_text[:3000]}"
                    "\n\nReturn JSON with this schema: {\n"
                    "  \"layers\": [\n"
                    "    {\"layer\": \"Context\", \"elements\": [string...], \"coverage\": float 0-1},\n"
                    "    {\"layer\": \"Drivers\", \"elements\": [string...], \"coverage\": float 0-1},\n"
                    "    {\"layer\": \"Metrics\", \"elements\": [string...], \"coverage\": float 0-1},\n"
                    "    {\"layer\": \"Scenarios\", \"elements\": [string...], \"coverage\": float 0-1}\n"
                    "  ],\n"
                    "  \"overall_coverage\": float 0-1,\n"
                    "  \"current_summary\": string,\n"
                    "  \"next_scenarios\": [{\"scenario\": string, \"confidence\": float 0-1}]\n"
                    "}"
                )
                try:
                    ai_text = gemini_chat([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ])
                except Exception as e:
                    ai_text = "{}"

                # Parse JSON robustly
                def parse_json_safe(text: str):
                    try:
                        return json.loads(text)
                    except Exception:
                        # Try to locate first/last braces
                        import re
                        m = re.search(r"\{[\s\S]*\}", text)
                        if m:
                            try:
                                return json.loads(m.group(0))
                            except Exception:
                                return {}
                        return {}

                parsed = parse_json_safe(ai_text) or {}
                layers = parsed.get("layers", [])
                # Fallback to expected layers if missing
                wanted = ["Context", "Drivers", "Metrics", "Scenarios"]
                if not layers:
                    layers = [{"layer": w, "elements": [], "coverage": 0.0} for w in wanted]
                # Normalize layer objects
                norm_layers = []
                for w in wanted:
                    found = next((l for l in layers if str(l.get("layer", "")).lower() == w.lower()), None)
                    if not found:
                        norm_layers.append({"layer": w, "elements": [], "coverage": 0.0})
                    else:
                        elems = found.get("elements", [])
                        if not isinstance(elems, list):
                            elems = [str(elems)]
                        cov = found.get("coverage", 0.0)
                        try:
                            cov = float(cov)
                        except Exception:
                            cov = 0.0
                        cov = max(0.0, min(1.0, cov))
                        norm_layers.append({"layer": w, "elements": [str(e) for e in elems], "coverage": cov})

                parsed_data = {"layers": norm_layers}
                overall_cov = parsed.get("overall_coverage")
                try:
                    coverage_score = float(overall_cov) * 100.0 if overall_cov is not None else (
                        sum(l["coverage"] for l in norm_layers) / max(len(norm_layers), 1) * 100.0
                    )
                except Exception:
                    coverage_score = 0.0

                scenario_json = {
                    "current": {"summary": parsed.get("current_summary", "")},
                    "next": parsed.get("next_scenarios", []) or []
                }

                # Store results in session to persist across reruns
                st.session_state.apply_results = {
                    "subject_id": subject_id,
                    "subject_name": selected_subject_name,
                    "narrative": narrative,
                    "parsed_data": parsed_data,
                    "coverage_score": coverage_score,
                    "scenario_json": scenario_json
                }
                st.rerun()

        # Render results from session (if present)
        results = st.session_state.get("apply_results")
        if results:
            parsed_data = results["parsed_data"]
            coverage_score = results["coverage_score"]
            scenario_json = results["scenario_json"]

            st.subheader("✅ Information Extraction Matrix")
            for i, layer in enumerate(parsed_data.get("layers", []), 1):
                with st.expander(f"Layer {i}: {layer.get('layer', 'Unknown')} (Coverage: {layer.get('coverage', 0)*100:.0f}%)"):
                    elements = layer.get('elements', [])
                    if elements:
                        for element in elements:
                            st.write(f"• {element}")
                    else:
                        st.write("No specific elements identified in this layer.")

            st.subheader("📊 Coverage Status")
            st.progress(min(max(int(coverage_score), 0), 100))
            st.write(f"Overall Coverage: {coverage_score:.1f}%")

            st.subheader("🧭 Scenario Mapping")
            current = scenario_json.get("current", {})
            if current:
                st.write("**Current Situation:**")
                st.write(current.get("summary", "No current summary available."))

            next_scenarios = scenario_json.get("next", [])
            if next_scenarios:
                st.write("**Potential Next Scenarios:**")
                for i, scenario in enumerate(next_scenarios, 1):
                    try:
                        confidence = float(scenario.get("confidence", 0)) * 100
                    except Exception:
                        confidence = 0
                    st.write(f"{i}. **{scenario.get('scenario', 'Unknown')}** (Confidence: {confidence:.0f}%)")

            # Save button (manual persist)
            if st.button("💾 Save Application", key="apply_save", type="primary"):
                record = {
                    "id": str(uuid.uuid4()),
                    "subject_id": results["subject_id"],
                    "input_text": results["narrative"],
                    "parsed_data": results["parsed_data"],
                    "coverage_score": results["coverage_score"],
                    "scenario_json": results["scenario_json"],
                    "created_at": datetime.datetime.now().isoformat()
                }
                try:
                    resp = supabase_client.table('subject_application').insert(record).execute()
                    if getattr(resp, 'data', None):
                        st.success("Application saved.")
                        # Clear inputs and outputs after save
                        st.session_state.apply_results = None
                        if 'apply_narrative' in st.session_state:
                            st.session_state['apply_narrative'] = ""
                        if 'apply_upload' in st.session_state:
                            st.session_state['apply_upload'] = None
                        # trigger a clean rerun to clear rendered output
                        st.rerun()
                    else:
                        st.warning("Insert returned no data; verify RLS and table permissions.")
                except Exception as e:
                    st.error(f"Failed to save application: {e}")

def render_ai_coach_ui(user_id: str):
    """Module 5: AI Coach on top of a selected application."""
    st.title("🧠 AI Coach for LeanChems")
    subjects_dict = get_all_subject_names()
    if not subjects_dict:
        st.warning("No subjects found. Please create a subject first.")
        return
    sorted_subject_names = sorted(subjects_dict.keys())
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        selected_subject_name = st.selectbox("Select Subject", options=sorted_subject_names, key="coach_subject_select")
        selected_app = None
        apps = []
        if selected_subject_name:
            subject_id = subjects_dict[selected_subject_name]
            try:
                resp = supabase_client.table('subject_application').select('id,created_at,coverage_score').eq('subject_id', subject_id).order('created_at', desc=True).execute()
                apps = resp.data or []
            except Exception as e:
                st.error(f"Failed to load applications: {e}")

            def _format_app_row(a: dict) -> str:
                # Human-readable label: Date • Coverage
                ts = a.get('created_at')
                label_date = str(ts)
                try:
                    # Handle both with/without timezone 'Z'
                    from datetime import datetime
                    label_date = datetime.fromisoformat(ts.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M') if ts else 'Unknown date'
                except Exception:
                    pass
                cov = a.get('coverage_score')
                try:
                    cov_str = f"{float(cov):.1f}%" if cov is not None else "n/a"
                except Exception:
                    cov_str = "n/a"
                return f"{label_date} • Coverage {cov_str}"

            selected_app = st.selectbox(
                "Select Previous Application",
                options=apps if apps else [None],
                format_func=(lambda a: _format_app_row(a) if isinstance(a, dict) else "None"),
                key="coach_app_select"
            )
        run = st.button("Analyze & Coach", key="coach_run", type="primary")
    with col2:
        if run and selected_app:
            # Placeholder coach logic; replace with real LLM analysis
            criticality_score = 72.0
            maturity_level = "Emerging"
            action_plan_json = {
                "short_term": ["Validate metrics", "Close top gaps"],
                "mid_term": ["Pilot scenario A", "Build dashboards"],
                "long_term": ["Institutionalize process", "Scale"]
            }
            recommendations_text = f"For {selected_subject_name}, focus on high impact/urgency gaps."

            # Keep results in session, do not auto-save
            st.session_state.coach_results = {
                "subject_application_id": selected_app['id'],
                "criticality_score": criticality_score,
                "maturity_level": maturity_level,
                "action_plan_json": action_plan_json,
                "recommendations_text": recommendations_text
            }
            st.rerun()

        # Render from session if available
        coach = st.session_state.get("coach_results")
        if coach:
            st.subheader("📈 Maturity & Criticality")
            st.metric("Criticality", f"{coach['criticality_score']:.1f}")
            st.write(f"Maturity: {coach['maturity_level']}")

            st.subheader("🧭 Recommended Strategy")
            plan = coach.get("action_plan_json", {})
            for section_key, section_title in [("short_term", "Short Term"), ("mid_term", "Mid Term"), ("long_term", "Long Term")]:
                items = plan.get(section_key, [])
                st.markdown(f"**{section_title}**")
                if items:
                    for i, itm in enumerate(items, 1):
                        st.write(f"{i}. {itm}")
                else:
                    st.write("No actions listed.")

            st.subheader("📝 Narrative Recommendations")
            st.write(coach.get("recommendations_text", ""))

            if st.button("💾 Save Coach Analysis", key="coach_save", type="primary"):
                record = {
                    "id": str(uuid.uuid4()),
                    "subject_application_id": coach['subject_application_id'],
                    "criticality_score": coach['criticality_score'],
                    "maturity_level": coach['maturity_level'],
                    "action_plan_json": coach['action_plan_json'],
                    "recommendations_text": coach['recommendations_text'],
                    "created_at": datetime.datetime.now().isoformat()
                }
                try:
                    resp = supabase_client.table('subject_coach_analysis').insert(record).execute()
                    if getattr(resp, 'data', None):
                        st.success("Coach analysis saved.")
                        # Clear session and selections
                        st.session_state.coach_results = None
                        st.rerun()
                    else:
                        st.warning("Insert returned no data; verify RLS and table permissions.")
                except Exception as e:
                    st.error(f"Failed to save coach analysis: {e}")

# Initialize session state
if not st.session_state.get("messages", None):
    st.session_state.messages = []

if not st.session_state.get("authenticated", None):
    st.session_state.authenticated = False

if not st.session_state.get("user", None):
    st.session_state.user = None

# Check for logout flag and clear it after processing
if st.session_state.get("logout_requested", False):
    st.session_state.logout_requested = False
    st.rerun()

# Initialize leanai_view state if it doesn't exist
if 'leanai_view' not in st.session_state:
    st.session_state.leanai_view = None

# Sidebar: Only login/logout/profile
with st.sidebar:
    st.sidebar.title("🤖 LeanAI Model MVP")
    # Add the logo here with error handling
 #   try:
   #     st.image("../leanchems logo.png", width=150)
   # except:
  #      st.markdown("### 🤖 LeanAI Model MVP")
    
    if not st.session_state.authenticated:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            st.subheader("Login")
            login_email = st.text_input("Email", key="sidebar_login_email")
            login_password = st.text_input("Password", type="password", key="sidebar_login_password")
            login_button = st.button("Login", key="sidebar_login_button", type="primary")
            if login_button:
                if login_email and login_password:
                    sign_in(login_email, login_password)
                else:
                    st.warning("Please enter both email and password.")
        with tab2:
            st.subheader("Sign Up")
            signup_name = st.text_input("Full Name", key="sidebar_signup_name")
            signup_email = st.text_input("Email", key="sidebar_signup_email")
            signup_password = st.text_input("Password", type="password", key="sidebar_signup_password")
            signup_button = st.button("Sign Up", key="sidebar_signup_button", type="primary")
            if signup_button:
                if signup_email and signup_password and signup_name:
                    response = sign_up(signup_email, signup_password, signup_name)
                    if response and response.user:
                        st.success("Sign up successful! Please check your email to confirm your account.")
                    else:
                        st.error("Sign up failed. Please try again.")
                else:
                    st.warning("Please fill in all fields.")
    else:
        user = st.session_state.user
        if user:
            # Get the user's full name from metadata, default to 'User' if not found
            full_name = user.user_metadata.get('full_name', 'User')
            # Display welcome message, centered and bold
            st.markdown(f"<div style='text-align: center;'><strong>Welcome, {full_name}!</strong></div>", unsafe_allow_html=True)
            st.button("Logout", on_click=sign_out, key="sidebar_logout_button", type="primary")

# Main dashboard for authenticated users
if st.session_state.authenticated and st.session_state.user:
    user_id = st.session_state.user.id
    
    if st.session_state.leanai_view is None:
        st.title("🤖 LeanAI Model MVP Dashboard")
        st.write("Build and manage your AI-powered knowledge base with intelligent document processing and narrative generation.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("📝 Create Subject", key="btn_create_subject", use_container_width=True, type="primary"):
                st.session_state.leanai_view = 'create'
                st.rerun()
        with col2:
            if st.button("📚 Manage MVP", key="btn_manage_mvp", use_container_width=True, type="primary"):
                st.session_state.leanai_view = 'manage'
                st.rerun()
        with col3:
            if st.button("📖 RAG for MVP", key="btn_rag_mvp", use_container_width=True, type="primary"):
                st.session_state.leanai_view = 'rag'
                st.rerun()
        with col4:
            if st.button("📌 Apply to LeanChems", key="btn_apply_mvp", use_container_width=True, type="primary"):
                st.session_state.leanai_view = 'apply'
                st.rerun()
        with col5:
            if st.button("🧠 AI Coach", key="btn_coach_mvp", use_container_width=True, type="primary"):
                st.session_state.leanai_view = 'coach'
                st.rerun()

    else:
        # Render the content of the selected view
        st.title("🤖 LeanAI Model MVP Dashboard")
        user_id = st.session_state.user.id

        # Add a button to go back to the initial selection view
        st.markdown("---")
       
        if st.button("🏠 Back to Main Menu", key="btn_back_to_menu"):
            st.session_state.leanai_view = None
            st.rerun()

        # Now render the section-specific content
        if st.session_state.leanai_view == 'create':
            render_subject_creation_ui_tab(user_id)
        elif st.session_state.leanai_view == 'manage':
            render_manage_subject_ui(user_id)
        elif st.session_state.leanai_view == 'rag':
            render_subject_rag_ui(user_id)
        elif st.session_state.leanai_view == 'apply':
            render_apply_to_leanchems_ui(user_id)
        elif st.session_state.leanai_view == 'coach':
            render_ai_coach_ui(user_id)

else:
    # Apply custom styling
    st.title("Welcome to LeanAI Model MVP")
    st.write("Build intelligent knowledge bases with AI-powered document processing and narrative generation.")
   
    # Feature highlights
    st.subheader("Features")
    col1, col2, col3 = st.columns(3)

    # Define Lottie animation URLs
    lottie_create = load_lottieurl("https://lottie.host/732c38ca-e084-4a94-a8c7-c2c1324b9700/WzO5X7h1a4.json")
    lottie_manage = load_lottieurl("https://lottie.host/embed/59a97f54-3a5c-494d-9d0f-b5504f6b308e/H6m7G94hUj.json")
    lottie_rag = load_lottieurl("https://lottie.host/8616b5b7-31f9-44c2-a110-76692c499332/oXqA8f8z07.json")

    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if lottie_create:
            st_lottie(lottie_create, height=150, key="create_animation")
        st.markdown("#### 📝 Create Subject Matter")
        st.write("Initialize new knowledge domains with AI-generated profiles and structured analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if lottie_manage:
            st_lottie(lottie_manage, height=150, key="manage_animation")
        st.markdown("#### 📚 Manage MVP Knowledge")
        st.write("Upload documents, add insights, and build comprehensive knowledge bases with intelligent chunking.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        if lottie_rag:
            st_lottie(lottie_rag, height=150, key="rag_animation")
        st.markdown("#### 📖 RAG-Powered Narratives")
        st.write("Generate comprehensive one-page narratives grounded in your knowledge base using advanced RAG.")
        st.markdown('</div>', unsafe_allow_html=True)

# Update the main execution block
if __name__ == "__main__":
    # This section won't run in Streamlit
    pass
