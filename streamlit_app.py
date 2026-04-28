import streamlit as st
import requests
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import torch
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings



# Access the API keys from secrets
#OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY

# Page configuration
st.set_page_config(
    page_title="SentimentSage - Financial Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Functions from your test.py (reused)
@st.cache_data
def fetch_news(query, api_key=NEWS_API_KEY):
    """Fetch financial news articles"""
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize=20&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return [f"{a['title']}. {a['description'] or ''}" for a in articles if a['title'] and a['description']]
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return []

@st.cache_resource
def get_finbert_pipeline():
    """Load FinBERT model for sentiment analysis"""
    try:
        model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading FinBERT model: {str(e)}")
        return None

def analyze_sentiment(articles, pipe):
    """Analyze sentiment of articles using FinBERT"""
    results = []
    progress_bar = st.progress(0)
    
    for i, text in enumerate(articles):
        try:
            sentiment = pipe(text[:512])[0]
            results.append({
                "text": text, 
                "label": sentiment['label'], 
                "score": sentiment['score']
            })
            progress_bar.progress((i + 1) / len(articles))
        except Exception as e:
            st.warning(f"Error analyzing article {i+1}: {str(e)}")
            continue
    
    progress_bar.empty()
    return results

def build_rag_pipeline(articles):
    """Build RAG pipeline for question answering"""
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("\n\n".join(articles))
            temp_file = f.name

        loader = TextLoader(temp_file, encoding="utf-8")
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        #embeddings = OpenAIEmbeddings()
        #db = FAISS.from_documents(docs, embeddings)

        # --- 新版 RAG 寫法 ---
        #llm = OpenAI(temperature=0)
        embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5") 
        
        db = FAISS.from_documents(docs, embeddings)

        # --- CHANGE 2: Use NVIDIA Chat Model ---
        # You can choose models like "meta/llama-3.1-405b-instruct" or "nvidia/nemotron-4-340b-instruct"
        llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0.2)
        
        # 定義提示詞模板
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided financial news context:
        <context>
        {context}
        </context>
        Question: {input}
        """)

        # 建立文件組合鏈
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        
        # 建立最終檢索鏈
        qa_chain = create_retrieval_chain(
            retriever=db.as_retriever(),
            combine_docs_chain=combine_docs_chain
        )

        os.unlink(temp_file)
        return qa_chain # 注意：新版回傳的是一個 Chain 物件
    except Exception as e:
        st.error(f"Error building RAG pipeline: {str(e)}")
        return None

def create_sentiment_chart(df):
    """Create sentiment distribution chart"""
    sentiment_counts = df['label'].value_counts()
    
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#ffc107'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_confidence_chart(df):
    """Create confidence score distribution"""
    fig = px.histogram(
        df, 
        x='score', 
        color='label',
        title="Confidence Score Distribution",
        nbins=20,
        color_discrete_map={
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#ffc107'
        }
    )
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">📈 Financial Sentiment Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Company input
        company = st.text_input(
            "Enter Company Ticker",
            placeholder="e.g., AAPL, TSLA, GOOGL",
            help="Enter the stock ticker symbol for the company you want to analyze"
        ).strip().upper()
        
        # Analysis options
        st.subheader("Analysis Options")
        include_charts = st.checkbox("Include Visualization Charts", value=True)
        include_rag = st.checkbox("Include AI Summary (RAG)", value=True)

        
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    # Main content area
    if not company:
        st.info("👈 Enter a company ticker in the sidebar to get started!")
        
        # Show example
        st.subheader("📋 How it works:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **1. Fetch News** 📰
            - Retrieves latest financial news
            - Uses NewsAPI for real-time data
            - Filters by relevance and recency
            """)
        
        with col2:
            st.markdown("""
            **2. Sentiment Analysis** 🎯
            - Uses FinBERT model
            - Specialized for financial text
            - Provides confidence scores
            """)
        
        with col3:
            st.markdown("""
            **3. AI Summary** 🤖
            - RAG-powered insights
            - Contextual analysis
            - Investment recommendations
            """)
        
        return
    
    if analyze_button:
        with st.spinner(f"Analyzing {company}..."):
            
            # Step 1: Fetch News
            st.subheader(f"📰 Latest News for {company}")
            articles = fetch_news(company)
            
            if not articles:
                st.error("No articles found. Please try a different ticker symbol.")
                return
            
            st.success(f"Found {len(articles)} articles")
            
            # Show sample articles
            with st.expander("📄 View Sample Articles"):
                for i, article in enumerate(articles[:3]):
                    st.write(f"**Article {i+1}:** {article[:200]}...")
            
            # Step 2: Sentiment Analysis
            st.subheader("🎯 Sentiment Analysis")
            
            finbert_pipeline = get_finbert_pipeline()
            if not finbert_pipeline:
                st.error("Failed to load FinBERT model")
                return
            
            with st.spinner("Analyzing sentiment..."):
                sentiment_results = analyze_sentiment(articles, finbert_pipeline)
            
            if not sentiment_results:
                st.error("Failed to analyze sentiment")
                return
            
            df = pd.DataFrame(sentiment_results)
            
            # Charts
            if include_charts:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = create_sentiment_chart(df)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    fig2 = create_confidence_chart(df)
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Detailed results table
            with st.expander("📊 Detailed Sentiment Results"):
                # Format the dataframe for better display
                display_df = df.copy()
                display_df['text'] = display_df['text'].str[:100] + "..."
                display_df['score'] = display_df['score'].round(3)
                
                # Color code sentiments
                def color_sentiment(val):
                    if val == 'positive':
                        return 'background-color: #d4edda'
                    elif val == 'negative':
                        return 'background-color: #f8d7da'
                    else:
                        return 'background-color: #fff3cd'
                
                styled_df = display_df.style.map(color_sentiment, subset=['label'])
                st.dataframe(styled_df, use_container_width=True)
            
            # Step 3: RAG Analysis
            if include_rag:
                st.subheader("🤖 AI-Powered Investment Summary")
                
                with st.spinner("Generating AI summary..."):
                    qa_pipeline = build_rag_pipeline([x["text"] for x in sentiment_results])
                    
                    if qa_pipeline:
                        # Use custom question or default
                        question = f"Can you provide a sentiment analysis and risk summary for {company} based on the news articles? Please give me a detailed analysis."
                        
                        try:
                            # CHANGE 1: Pass the question as a dictionary with the key "input"
                            result = qa_pipeline.invoke({"input": question})
                            
                            st.markdown("### 📝 Summary")
                            # CHANGE 2: The new chain outputs the text under 'answer', not 'result'
                            st.write(result['answer'])
                            
                            # CHANGE 3: The source documents are now under 'context', not 'source_documents'
                            if 'context' in result and result['context']:
                                with st.expander("📚 Source Documents"):
                                    for i, doc in enumerate(result['context'][:3]):
                                        st.write(f"**Source {i+1}:** {doc.page_content[:300]}...")
                        
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                    else:
                        st.error("Failed to build RAG pipeline")
            

if __name__ == "__main__":
    main()
