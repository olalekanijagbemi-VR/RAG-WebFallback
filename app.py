"""
RAG-WebFallback - Main Application (FINAL)
Multi-Agent RAG System with Web Fallback
Apple Liquid Glass UI - Dark Bars - 3D Italic Title - Big Bold Input
"""

import os
import sys
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import our modules
from src.document_processor import DocumentProcessor
from src.hybrid_search import HybridSearch
from src.router_agent import RouterAgent
from src.web_fallback import WebFallback
from src.answer_generator import AnswerGenerator

# Page configuration
st.set_page_config(
    page_title="RAG-WebFallback",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# APPLE LIQUID GLASS UI - DARK BARS - 3D ITALIC TITLE
# FIXED: Input field visible with Apple Liquid Glass style
# ============================================================
st.markdown("""
<style>
    /* ============================================================
       BACKGROUND: High Resolution Reflective Metal at 75% intensity
       ============================================================ */
    
    .stApp {
        background-color: #4a4a4a;
        
        /* High Resolution Metal Texture */
        background-image: 
            repeating-linear-gradient(90deg, 
                rgba(0,0,0,0.02) 0px, 
                rgba(255,255,255,0.02) 0.5px, 
                transparent 1px, 
                transparent 3px,
                rgba(0,0,0,0.015) 3px,
                rgba(255,255,255,0.015) 3.5px,
                transparent 4px,
                transparent 6px
            ),
            repeating-linear-gradient(90deg, 
                rgba(0,0,0,0.03) 0px, 
                transparent 2px, 
                rgba(255,255,255,0.02) 4px, 
                transparent 6px
            ),
            linear-gradient(90deg, 
                #3a3a3a 0%, 
                #6a6a6a 25%, 
                #b0b0b0 40%, 
                #cccccc 50%, 
                #b0b0b0 60%, 
                #6a6a6a 75%, 
                #3a3a3a 100%
            );
            
        background-blend-mode: overlay, overlay, normal;
        background-size: cover, cover, cover;
        background-attachment: fixed;
    }

    /* ============================================================
       TOP SEAM - Double Width (12px) - Dark
       ============================================================ */
    
    .stApp::before {
        content: "";
        position: fixed;
        top: 10%;
        left: 0;
        width: 100%;
        height: 12px;
        z-index: 9999;
        pointer-events: none;
        background: 
            linear-gradient(to bottom, 
                rgba(0,0,0,0.8) 0px, 
                rgba(0,0,0,0.95) 3px, 
                rgba(60,50,40,0.4) 4px,
                rgba(180,170,160,0.3) 5px,
                rgba(255,255,255,0.6) 6px, 
                rgba(255,255,255,0.3) 7px,
                rgba(180,170,160,0.15) 8px,
                transparent 12px
            );
    }

    /* ============================================================
       BOTTOM SEAM - Double Width (12px) - Dark
       ============================================================ */
    
    .stApp::after {
        content: "";
        position: fixed;
        bottom: 10%;
        left: 0;
        width: 100%;
        height: 12px;
        z-index: 9999;
        pointer-events: none;
        background: 
            linear-gradient(to bottom, 
                transparent 0px,
                rgba(180,170,160,0.15) 4px,
                rgba(255,255,255,0.3) 5px,
                rgba(255,255,255,0.6) 6px, 
                rgba(180,170,160,0.3) 7px,
                rgba(60,50,40,0.4) 8px,
                rgba(0,0,0,0.95) 9px, 
                rgba(0,0,0,0.8) 12px
            );
    }

    /* ============================================================
       TOP BAR - DARK BLACK (NO WHITE, NO BLUE)
       ============================================================ */
    
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0.95) !important;
        backdrop-filter: none !important;
        border-bottom: 1px solid rgba(255,255,255,0.05) !important;
        box-shadow: 0px 2px 20px rgba(0,0,0,0.8) !important;
        height: 48px !important;
        min-height: 48px !important;
    }
    
    .stApp > header {
        background: rgba(0,0,0,0.95) !important;
    }
    
    .st-emotion-cache-1r6slb0 {
        background: rgba(0,0,0,0.95) !important;
    }

    /* ============================================================
       HIDE FOOTER ONLY - KEEP CHAT INPUT
       ============================================================ */

    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    
    #MainMenu {
        display: none !important;
        visibility: hidden !important;
    }
    
    .st-emotion-cache-1r6slb0 {
        display: none !important;
    }

    /* ============================================================
       CUSTOM BLACK BOTTOM BAR (THIN, MATCHES TOP)
       ============================================================ */

    body::after {
        content: "";
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 48px;
        z-index: 999999;
        pointer-events: none;
        background: linear-gradient(180deg, 
            #1a1a1a 0%, 
            #2a2a2a 25%, 
            #1a1a1a 50%, 
            #111111 75%, 
            #0a0a0a 100%
        );
        background-image: 
            radial-gradient(ellipse at 5% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 15% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 25% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 35% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 45% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 55% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 65% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 75% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 85% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            radial-gradient(ellipse at 95% 50%, rgba(50,50,50,0.4) 2px, rgba(30,30,30,0.2) 3px, transparent 4px),
            repeating-linear-gradient(90deg, 
                rgba(0,0,0,0.05) 0px, 
                rgba(255,255,255,0.02) 1px, 
                transparent 2px, 
                transparent 6px
            );
        background-blend-mode: overlay, overlay, normal;
        background-size: auto, cover;
        background-repeat: repeat-x, no-repeat;
        border-top: 2px solid rgba(0,0,0,0.5);
        box-shadow: 0px -2px 10px rgba(0,0,0,0.6);
    }

    /* ============================================================
       MAIN TITLE - 3D Effect, Shadow, Slightly Italic
       Gear Icon stays STRAIGHT (not italic)
       ============================================================ */
    
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        color: #809559 !important;
        letter-spacing: 2px;
        margin-bottom: 0.2rem;
        padding: 0.5rem 0;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    
    .main-header .gear-icon {
        font-style: normal !important;
        font-weight: 400;
        display: inline-block;
        transform: none !important;
        text-shadow: 
            0px 2px 4px rgba(0,0,0,0.2),
            0px 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    .main-header .title-text {
        font-style: italic !important;
        text-shadow: 
            0px 1px 0px rgba(0,0,0,0.1),
            0px 2px 0px rgba(0,0,0,0.15),
            0px 3px 0px rgba(0,0,0,0.2),
            0px 4px 0px rgba(0,0,0,0.25),
            0px 5px 0px rgba(0,0,0,0.3),
            0px 6px 0px rgba(0,0,0,0.35),
            0px 8px 12px rgba(0,0,0,0.5),
            0px 12px 24px rgba(0,0,0,0.3) !important;
        transform: skewX(-3deg);
        display: inline-block;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #1a1a1a !important;
        text-shadow: 0px 2px 8px rgba(0,0,0,0.3) !important;
        margin-bottom: 2rem;
        opacity: 0.85;
        font-weight: 500;
    }

    /* ============================================================
       RESPONSE TEXT - Very Dark and BOLD
       ============================================================ */
    
    .stChatMessage div, 
    .stChatMessage p, 
    .stChatMessage span,
    .stChatMessage .stMarkdown {
        color: #0a0a0a !important;
        font-weight: 700 !important;
        text-shadow: 
            0px 1px 2px rgba(255, 255, 255, 0.08),
            0px 2px 8px rgba(255, 255, 255, 0.03) !important;
        letter-spacing: 0.01em;
    }
    
    .stChatMessage .stMarkdown p {
        color: #0a0a0a !important;
        font-weight: 700 !important;
    }

    /* ============================================================
       APPLE LIQUID GLASS - Response Layer
       ============================================================ */
    
    .stChatMessage[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"]:nth-child(1)) {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 1.2rem !important;
        padding: 1rem 1.5rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 
            0px 4px 24px rgba(0, 0, 0, 0.08),
            0px 1px 0px rgba(255, 255, 255, 0.3) inset !important;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent"]:nth-child(2)) {
        background: rgba(220, 220, 235, 0.12) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 1.2rem !important;
        padding: 1rem 1.5rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 
            0px 4px 24px rgba(0, 0, 0, 0.06),
            0px 1px 0px rgba(255, 255, 255, 0.2) inset !important;
    }

    /* ============================================================
       CHAT INPUT - APPLE LIQUID GLASS STYLE (VISIBLE)
       ============================================================ */
    
    .stChatInput {
        position: fixed !important;
        bottom: 60px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 80% !important;
        max-width: 800px !important;
        z-index: 999999 !important;
        padding: 0 !important;
    }
    
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 1.2rem !important;
        box-shadow: 
            0px 4px 30px rgba(0, 0, 0, 0.15),
            0px 1px 0px rgba(255, 255, 255, 0.3) inset !important;
        transition: all 0.3s ease !important;
        padding: 0.25rem !important;
    }
    
    .stChatInput > div:focus-within {
        background: rgba(255, 255, 255, 0.22) !important;
        border-color: rgba(255, 255, 255, 0.4) !important;
        box-shadow: 
            0px 4px 40px rgba(0, 0, 0, 0.2),
            0px 1px 0px rgba(255, 255, 255, 0.4) inset !important;
    }
    
    .stChatInput input {
        color: #0a0a0a !important;
        background: transparent !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        text-shadow: 
            0px 1px 2px rgba(255, 255, 255, 0.1) !important;
        letter-spacing: 0.02em;
        padding: 0.75rem 1.2rem !important;
        height: 56px !important;
    }
    
    .stChatInput input::placeholder {
        color: rgba(0, 0, 0, 0.35) !important;
        opacity: 0.8;
        font-weight: 400;
        font-size: 1rem !important;
        text-shadow: none !important;
    }

    /* ============================================================
       SIDEBAR - KEEP ORIGINAL (White text, glass effect)
       ============================================================ */
    
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 20, 0.7) !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        z-index: 99999 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label {
        color: #ffffff !important;
        text-shadow: 0px 2px 8px rgba(0,0,0,0.9) !important;
    }

    /* ============================================================
       SOURCE BOX - Apple Liquid Glass style
       ============================================================ */
    
    .source-box {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding: 0.6rem 1rem;
        border-radius: 0.75rem;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #0a0a0a !important;
        text-shadow: none !important;
        box-shadow: 0px 2px 12px rgba(0,0,0,0.06);
    }
    
    .source-box b {
        color: #0a0a0a !important;
    }
    
    .source-box span {
        color: #0a0a0a !important;
        text-shadow: none !important;
    }

    /* ============================================================
       CONFIDENCE COLORS - Deep Bold Colors
       ============================================================ */
    
    .confidence-high {
        color: #0a6e1a !important;
        font-weight: 700;
        text-shadow: none !important;
    }
    
    .confidence-medium {
        color: #8a6d00 !important;
        font-weight: 700;
        text-shadow: none !important;
    }
    
    .confidence-low {
        color: #8a1a1a !important;
        font-weight: 700;
        text-shadow: none !important;
    }

    /* ============================================================
       BUTTONS - Dark Matte
       ============================================================ */
    
    .stButton > button {
        background: linear-gradient(180deg, #2a2a2a, #1a1a1a) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        color: #ffffff !important;
        text-shadow: 0px 1px 4px rgba(0,0,0,0.8) !important;
        border-radius: 0.75rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0px 2px 12px rgba(0,0,0,0.4) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(180deg, #3a3a3a, #2a2a2a) !important;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.5) !important;
        transform: translateY(-1px);
        border-color: rgba(255,255,255,0.12) !important;
    }

    /* ============================================================
       EXPANDER - Apple Liquid Glass style
       ============================================================ */
    
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 0.75rem !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #0a0a0a !important;
        text-shadow: none !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border-radius: 0 0 0.75rem 0.75rem !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-top: none !important;
    }

    /* ============================================================
       FILE UPLOADER - Apple Liquid Glass style
       ============================================================ */
    
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 0.75rem !important;
        color: #0a0a0a !important;
    }

    /* ============================================================
       METRIC CARDS - Apple Liquid Glass style
       ============================================================ */
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border-radius: 0.75rem !important;
        padding: 0.5rem 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        box-shadow: 0px 2px 12px rgba(0,0,0,0.06);
    }
    
    .stMetric label {
        color: #0a0a0a !important;
        text-shadow: none !important;
    }
    
    .stMetric .stMetricValue {
        color: #0a0a0a !important;
        text-shadow: none !important;
    }

    /* ============================================================
       DIVIDERS - Subtle
       ============================================================ */
    
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, 
            transparent 0%, 
            rgba(0, 0, 0, 0.08) 30%, 
            rgba(0, 0, 0, 0.12) 50%, 
            rgba(0, 0, 0, 0.08) 70%, 
            transparent 100%
        ) !important;
        margin: 1.5rem 0 !important;
    }

    /* ============================================================
       SCROLLBAR - Dark
       ============================================================ */
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 30, 0.4) !important;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(80, 80, 80, 0.5) !important;
        border-radius: 4px;
        border: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(80, 80, 80, 0.7) !important;
    }

    /* ============================================================
       ALERT MESSAGES - Apple Liquid Glass style
       ============================================================ */
    
    .stAlert {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 0.75rem !important;
        color: #0a0a0a !important;
        text-shadow: none !important;
    }
    
    .stAlert .stMarkdown,
    .stAlert div, 
    .stAlert p, 
    .stAlert span {
        color: #0a0a0a !important;
        text-shadow: none !important;
    }

    /* ============================================================
       LINKS - Deep Black
       ============================================================ */
    
    .stMarkdown a {
        color: #0a0a0a !important;
        text-decoration: underline;
        text-underline-offset: 2px;
        text-shadow: none !important;
    }
    
    .stMarkdown a:hover {
        color: #333333 !important;
    }

    /* ============================================================
       BLOCKQUOTE / STREAMLIT DEFAULT OVERRIDES - DARK
       ============================================================ */
    
    .st-emotion-cache-1r6slb0 {
        background: rgba(0,0,0,0.95) !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0.95) !important;
    }
    
    .st-emotion-cache-12fmjuu {
        background: rgba(0,0,0,0.95) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'document_count' not in st.session_state:
        st.session_state.document_count = 0
    if 'chunk_count' not in st.session_state:
        st.session_state.chunk_count = 0
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []

# Initialize components
@st.cache_resource
def init_components():
    """Initialize all components (cached for performance)"""
    return {
        'document_processor': DocumentProcessor(),
        'router_agent': RouterAgent(),
        'web_fallback': WebFallback(max_results=5),
        'answer_generator': AnswerGenerator(),
        'hybrid_search': HybridSearch()
    }

# Main UI
def main():
    """Main application"""
    
    # Initialize
    init_session_state()
    components = init_components()
    
    # Header - Gear icon STRAIGHT, Title ITALIC
    st.markdown(
        """
        <div class="main-header">
            <span class="gear-icon">⚙️</span>
            <span class="title-text">RAG-WebFallback</span>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown('<div class="sub-header">Multi-Agent RAG System with Web Fallback • Source Tracking • Confidence Scoring</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, DOCX, or CSV files",
            type=['pdf', 'txt', 'docx', 'csv'],
            accept_multiple_files=True,
            help="Upload multiple documents for the RAG system to process"
        )
        
        # Process button
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        all_chunks = components['document_processor'].process_multiple_files(uploaded_files)
                        
                        if all_chunks:
                            # Build hybrid search index
                            success = components['hybrid_search'].build_index(all_chunks)
                            
                            if success:
                                st.session_state.vectorstore = components['hybrid_search']
                                st.session_state.documents_loaded = True
                                st.session_state.document_count = len(uploaded_files)
                                st.session_state.chunk_count = len(all_chunks)
                                st.session_state.processed_documents = [f.name for f in uploaded_files]
                                
                                st.success(f"✅ Processed {len(uploaded_files)} files with {len(all_chunks)} chunks!")
                            else:
                                st.error("❌ Failed to build search index")
                        else:
                            st.warning("⚠️ No chunks created from uploaded files")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing documents: {str(e)}")
        
        # Document status
        st.divider()
        st.subheader("📊 System Status")
        
        if st.session_state.documents_loaded:
            st.success(f"✅ {st.session_state.document_count} documents loaded")
            st.info(f"📑 {st.session_state.chunk_count} total chunks")
            
            # Show processed files
            with st.expander("📂 Processed Files"):
                for doc in st.session_state.processed_documents:
                    st.write(f"• {doc}")
        else:
            st.warning("⚠️ No documents loaded")
        
        # Router status
        st.divider()
        st.subheader("🤖 Router Status")
        st.info("Smart routing between documents and web")
        
        # Clear chat button
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Version
        st.divider()
        st.caption("v1.0.0 • Built with Streamlit + Groq + FAISS")

    # Main chat area
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                
                # Show sources if available
                if 'sources' in message and message['sources']:
                    with st.expander(f"📚 View Sources ({len(message['sources'])})"):
                        for i, source in enumerate(message['sources'], 1):
                            source_type = source.get('source_type', 'unknown').capitalize()
                            source_name = source.get('source', source.get('title', 'Unknown'))
                            confidence = source.get('confidence', source.get('relevance', 0))
                            
                            # Confidence color
                            if confidence > 0.7:
                                conf_class = "confidence-high"
                            elif confidence > 0.4:
                                conf_class = "confidence-medium"
                            else:
                                conf_class = "confidence-low"
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <b>Source {i}</b> ({source_type}) - 
                                <span class="{conf_class}">Confidence: {confidence:.2%}</span><br>
                                <span style="font-size:0.85rem; color:#333333;">{source_name}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show content preview
                            content = source.get('content', '')[:200]
                            if content:
                                st.caption(f"📝 {content}...")
                            
                            if source.get('url'):
                                st.write(f"🔗 [Link]({source['url']})")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents or current events..."):
        # Add user message
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # 1. Route the query
                    has_docs = st.session_state.documents_loaded
                    router_decision = components['router_agent'].get_route_decision(prompt, has_docs)
                    
                    # 2. Get results based on route
                    sources = []
                    context = ""
                    used_web = False
                    used_docs = False
                    
                    # First, try documents if available
                    if has_docs:
                        search_results = components['hybrid_search'].search_with_context(prompt, k=5)
                        doc_sources = search_results.get('sources', [])
                        
                        # Check if document results are actually relevant
                        doc_results = search_results.get('results', [])
                        should_use_web = components['router_agent'].should_use_web_based_on_results(
                            prompt, 
                            doc_results,
                            has_docs
                        )
                        
                        # Check if documents have any relevant content
                        has_relevant_docs = False
                        if doc_sources:
                            for src in doc_sources[:3]:
                                if src.get('confidence', 0) > 0.5:
                                    has_relevant_docs = True
                                    break
                        
                        if doc_sources and not should_use_web and has_relevant_docs:
                            # Document results are good - use them
                            sources.extend(doc_sources)
                            context = search_results.get('context', '')
                            used_docs = True
                            print(f"📚 Using {len(sources)} document sources")
                        else:
                            # Document results are not relevant - use web
                            print("📚 Document results not relevant, using web search")
                            web_results = components['web_fallback'].search_with_sources(prompt, max_results=5)
                            if web_results['sources']:
                                sources.extend(web_results['sources'])
                                context = web_results['context']
                                used_web = True
                                print(f"🌐 Using {len(sources)} web sources")
                    
                    # If no sources found yet, try web as final fallback
                    if not sources:
                        print("🌐 No relevant documents, using web search")
                        web_results = components['web_fallback'].search_with_sources(prompt, max_results=5)
                        if web_results['sources']:
                            sources.extend(web_results['sources'])
                            context = web_results['context']
                            used_web = True
                            print(f"🌐 Using {len(sources)} web sources")
                    
                    # If STILL no sources, try a more general web search
                    if not sources:
                        print("🌐 Trying general web search...")
                        web_results = components['web_fallback'].search_with_sources(prompt, max_results=3)
                        if web_results['sources']:
                            sources.extend(web_results['sources'])
                            context = web_results['context']
                            used_web = True
                            print(f"🌐 Using {len(sources)} web sources (general search)")
                    
                    # 3. Generate answer
                    if sources and context:
                        answer_result = components['answer_generator'].generate_answer(prompt, context, sources)
                        answer = answer_result['answer']
                        cited_sources = answer_result.get('cited_sources', sources)
                    else:
                        answer = "I couldn't find relevant information to answer your question. Please try rephrasing or upload more documents."
                        cited_sources = []
                    
                    # 4. Display answer
                    st.markdown(answer)
                    
                    # Display route info
                    with st.expander("🤔 Routing Decision"):
                        st.write(f"**Route:** {router_decision['route'].upper()}")
                        st.write(f"**Confidence:** {router_decision['confidence']:.2%}")
                        st.write(f"**Reasoning:** {router_decision['reasoning']}")
                        if router_decision['keywords_matched']:
                            st.write(f"**Keywords matched:** {', '.join(router_decision['keywords_matched'])}")
                        if used_web:
                            st.write("**🌐 Web search was used** (documents didn't have relevant info)")
                        elif used_docs and sources:
                            st.write("**📚 Used documents** (found relevant information)")
                        elif not sources:
                            st.write("**⚠️ No sources found** (try rephrasing your question)")
                    
                    # Display sources
                    if cited_sources:
                        with st.expander(f"📚 View Sources ({len(cited_sources)})"):
                            for i, source in enumerate(cited_sources, 1):
                                source_type = source.get('source_type', 'unknown').capitalize()
                                source_name = source.get('source', source.get('title', 'Unknown'))
                                confidence = source.get('confidence', source.get('relevance', 0))
                                content = source.get('content', '')[:300]
                                
                                st.markdown(f"""
                                <div class="source-box">
                                    <b>Source {i}</b> ({source_type}) - 
                                    Confidence: {confidence:.2%}<br>
                                    <span style="font-size:0.85rem; color:#333333;">{source_name}</span>
                                </div>
                                """, unsafe_allow_html=True)
                                st.caption(f"📝 {content}...")
                                if source.get('url'):
                                    st.write(f"🔗 [Link]({source['url']})")
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    answer = f"I encountered an error while processing your question. Please try again."
                    st.markdown(answer)
        
        # Save to chat history
        st.session_state.messages.append({
            'role': 'assistant',
            'content': answer,
            'sources': cited_sources
        })

if __name__ == "__main__":
    main()