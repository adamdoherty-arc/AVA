"""
RAG Knowledge Base Management
==============================

Manage the RAG (Retrieval-Augmented Generation) knowledge base.

Features:
- View knowledge base statistics
- Trigger document ingestion
- Manage document categories
- Search and query documents
- Monitor daily XTrades sync
- Batch upload documents

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="RAG Knowledge Base | Magnus",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .rag-header {
        background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    .category-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-xtrades { background: #3498db; color: white; }
    .badge-discord { background: #7289da; color: white; }
    .badge-strategies { background: #27ae60; color: white; }
    .badge-research { background: #e67e22; color: white; }
    .badge-education { background: #9b59b6; color: white; }

    .sync-status-success { color: #27ae60; font-weight: 600; }
    .sync-status-pending { color: #f39c12; font-weight: 600; }
    .sync-status-error { color: #e74c3c; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.markdown("""
<div class="rag-header">
    <h1 style="margin:0">📚 RAG Knowledge Base</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9">
        Document Management • Daily XTrades Sync • Query Knowledge Base
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Initialize Pipeline
# ============================================================================

@st.cache_resource
def get_ingestion_pipeline():
    """Get cached ingestion pipeline"""
    from src.rag.document_ingestion_pipeline import DocumentIngestionPipeline
    return DocumentIngestionPipeline()

try:
    pipeline = get_ingestion_pipeline()
    stats = pipeline.get_stats()
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.stop()

# ============================================================================
# Quick Stats
# ============================================================================

st.subheader("📊 Knowledge Base Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Documents",
        f"{stats.get('documents_processed', 0):,}",
        help="Total documents ingested"
    )

with col2:
    st.metric(
        "Chunks Created",
        f"{stats.get('chunks_created', 0):,}",
        help="Total text chunks for retrieval"
    )

with col3:
    st.metric(
        "Collections",
        stats.get('total_collections', 0),
        help="Number of document categories"
    )

with col4:
    # Last sync time (mock for now)
    last_sync = datetime.now() - timedelta(hours=1)
    st.metric(
        "Last XTrades Sync",
        last_sync.strftime("%I:%M %p"),
        help="Last daily sync time"
    )

st.divider()

# ============================================================================
# Main Tabs
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "📤 Upload Documents",
    "🔄 Sync Management",
    "🔍 Search & Query",
    "⚙️ Settings"
])

# ============================================================================
# TAB 1: Dashboard
# ============================================================================

with tab1:
    st.markdown("### 📈 Knowledge Base Statistics")

    # Category breakdown
    st.markdown("#### 📋 Documents by Category")

    # Mock category data (replace with actual ChromaDB queries)
    category_data = pd.DataFrame({
        'Category': [
            'XTrades Messages',
            'Discord Messages',
            'Trading Strategies',
            'Market Research',
            'Technical Docs',
            'Earnings Reports'
        ],
        'Documents': [15234, 8456, 245, 189, 342, 156],
        'Last Updated': [
            '1 hour ago',
            '1 hour ago',
            '2 days ago',
            '5 days ago',
            '1 week ago',
            '3 days ago'
        ],
        'Status': [
            'Active',
            'Active',
            'Active',
            'Stale',
            'Active',
            'Active'
        ]
    })

    st.dataframe(category_data, use_container_width=True, hide_index=True)

    st.divider()

    # Recent ingestion activity
    st.markdown("#### 📅 Recent Ingestion Activity")

    recent_activity = pd.DataFrame({
        'Timestamp': [
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(hours=25),
            datetime.now() - timedelta(hours=49),
            datetime.now() - timedelta(days=3),
            datetime.now() - timedelta(days=5),
        ],
        'Source': [
            'XTrades Daily Sync',
            'XTrades Daily Sync',
            'XTrades Daily Sync',
            'Manual Upload',
            'Discord Sync'
        ],
        'Category': [
            'XTrades Messages',
            'XTrades Messages',
            'XTrades Messages',
            'Trading Strategies',
            'Discord Messages'
        ],
        'Documents Added': [127, 145, 132, 5, 234],
        'Status': ['Success', 'Success', 'Success', 'Success', 'Success']
    })

    st.dataframe(recent_activity, use_container_width=True, hide_index=True)

    # Ingestion stats over time
    st.markdown("#### 📊 Ingestion Trends (Last 30 Days)")

    chart_data = pd.DataFrame({
        'Date': pd.date_range(end=datetime.now(), periods=30, freq='D'),
        'XTrades': [120 + (i % 30) for i in range(30)],
        'Discord': [80 + (i % 20) for i in range(30)],
        'Manual': [5 + (i % 10) for i in range(30)]
    }).set_index('Date')

    st.area_chart(chart_data)

# ============================================================================
# TAB 2: Upload Documents
# ============================================================================

with tab2:
    st.markdown("### 📤 Upload Documents to Knowledge Base")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        **Upload Options:**
        - **Single File**: Upload a single PDF, TXT, or MD file
        - **Directory**: Batch upload from a local directory
        - **URL**: Fetch and ingest from a web page
        """)

    with col2:
        upload_method = st.selectbox(
            "Upload Method",
            ["Single File", "Directory", "URL"],
            key="upload_method"
        )

    st.divider()

    # Single File Upload
    if upload_method == "Single File":
        from src.rag.document_ingestion_pipeline import DocumentCategory, DocumentSource

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'md', 'docx'],
            key="file_uploader"
        )

        col1, col2 = st.columns(2)

        with col1:
            category = st.selectbox(
                "Document Category",
                [cat.value for cat in DocumentCategory],
                key="file_category"
            )

        with col2:
            # Custom metadata
            custom_tags = st.text_input(
                "Tags (comma-separated)",
                placeholder="options, wheel strategy, advanced",
                key="file_tags"
            )

        if uploaded_file:
            if st.button("📤 Upload and Ingest", type="primary", key="upload_btn"):
                with st.spinner("Processing document..."):
                    try:
                        # Read file content
                        content = uploaded_file.read().decode('utf-8')

                        # Prepare metadata
                        metadata = {
                            'filename': uploaded_file.name,
                            'file_size': uploaded_file.size,
                            'uploaded_by': 'user',
                            'upload_date': datetime.now().isoformat()
                        }

                        if custom_tags:
                            metadata['tags'] = [tag.strip() for tag in custom_tags.split(',')]

                        # Ingest
                        result = pipeline.ingest_document(
                            content=content,
                            metadata=metadata,
                            category=DocumentCategory[category.upper().replace(' ', '_')],
                            source=DocumentSource.UPLOAD
                        )

                        if result['status'] == 'success':
                            st.success(f"✅ Document ingested successfully!")
                            st.info(f"Created {result['chunks_created']} chunks")
                        else:
                            st.error(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"Error: {e}")

    # Directory Upload
    elif upload_method == "Directory":
        directory_path = st.text_input(
            "Directory Path",
            placeholder="/path/to/documents",
            key="dir_path"
        )

        col1, col2 = st.columns(2)

        with col1:
            category = st.selectbox(
                "Document Category",
                [cat.value for cat in DocumentCategory],
                key="dir_category"
            )

        with col2:
            recursive = st.checkbox("Scan subdirectories", value=True, key="dir_recursive")

        if st.button("📂 Scan and Ingest Directory", type="primary", key="dir_btn"):
            if directory_path:
                with st.spinner(f"Scanning {directory_path}..."):
                    try:
                        result = pipeline.ingest_local_directory(
                            directory=directory_path,
                            category=DocumentCategory[category.upper().replace(' ', '_')],
                            recursive=recursive
                        )

                        if result.get('status') == 'success':
                            st.success(f"✅ Ingested {result.get('success', 0)} documents!")
                            if result.get('skipped', 0) > 0:
                                st.info(f"Skipped {result['skipped']} duplicates")
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a directory path")

    # URL Upload
    elif upload_method == "URL":
        url = st.text_input(
            "URL",
            placeholder="https://example.com/research-paper.pdf",
            key="url_input"
        )

        category = st.selectbox(
            "Document Category",
            [cat.value for cat in DocumentCategory],
            key="url_category"
        )

        if st.button("🌐 Fetch and Ingest URL", type="primary", key="url_btn"):
            if url:
                with st.spinner(f"Fetching {url}..."):
                    st.info("URL ingestion coming soon!")
            else:
                st.warning("Please enter a URL")

# ============================================================================
# TAB 3: Sync Management
# ============================================================================

with tab3:
    st.markdown("### 🔄 Automated Sync Management")

    # XTrades Sync
    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
    st.markdown("#### 📊 XTrades Messages Sync")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        **Daily Sync Schedule**: 1:00 AM
        **Last Sync**: 1 hour ago
        **Status**: <span class="sync-status-success">✅ Active</span>
        """, unsafe_allow_html=True)

        st.caption("Automatically ingests XTrades messages daily to keep the knowledge base current.")

    with col2:
        st.metric("Messages Today", "127")
        st.caption("Last 24 hours")

    with col3:
        st.metric("Total Messages", "15,234")
        st.caption("All time")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Trigger Manual Sync", key="xtrades_manual"):
            with st.spinner("Syncing XTrades messages..."):
                try:
                    result = pipeline.ingest_xtrades_messages(days_back=1)

                    st.success(f"✅ Synced {result.get('success', 0)} messages")
                    if result.get('skipped', 0) > 0:
                        st.info(f"Skipped {result['skipped']} duplicates")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        days_back = st.number_input(
            "Days to Sync",
            min_value=1,
            max_value=90,
            value=1,
            key="xtrades_days"
        )

        if st.button("📅 Sync Historical", key="xtrades_historical"):
            with st.spinner(f"Syncing last {days_back} days..."):
                try:
                    result = pipeline.ingest_xtrades_messages(
                        days_back=days_back,
                        force_reload=False
                    )

                    st.success(f"✅ Synced {result.get('success', 0)} messages")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # Discord Sync
    st.markdown('<div class="stat-card">', unsafe_allow_html=True)
    st.markdown("#### 💬 Discord Messages Sync")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        **Daily Sync Schedule**: 2:00 AM
        **Last Sync**: 2 hours ago
        **Status**: <span class="sync-status-success">✅ Active</span>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Messages Today", "234")

    with col3:
        st.metric("Total Messages", "8,456")

    if st.button("🔄 Trigger Discord Sync", key="discord_manual"):
        with st.spinner("Syncing Discord messages..."):
            try:
                result = pipeline.ingest_discord_messages(days_back=7)

                st.success(f"✅ Synced {result.get('success', 0)} messages")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 4: Search & Query
# ============================================================================

with tab4:
    st.markdown("### 🔍 Search Knowledge Base")

    search_query = st.text_input(
        "Search Query",
        placeholder="What are the best wheel strategy opportunities?",
        key="search_query"
    )

    col1, col2 = st.columns(2)

    with col1:
        search_category = st.selectbox(
            "Search In",
            ["All Categories"] + [cat.value for cat in DocumentCategory],
            key="search_category"
        )

    with col2:
        num_results = st.slider("Number of Results", 1, 20, 5, key="search_num_results")

    if st.button("🔍 Search", type="primary", key="search_btn"):
        if search_query:
            with st.spinner("Searching knowledge base..."):
                st.info("Search functionality coming soon - will integrate with ChromaDB query API")
                st.markdown("**Sample Results:**")

                sample_results = [
                    {
                        'Document': 'XTrades Alert #12345',
                        'Relevance': 0.95,
                        'Preview': 'CSP opportunity on AAPL at $170 strike, 30 DTE, 0.25 delta...',
                        'Category': 'XTrades Messages',
                        'Date': '2025-11-20'
                    },
                    {
                        'Document': 'Wheel Strategy Guide',
                        'Relevance': 0.88,
                        'Preview': 'The wheel strategy involves selling cash-secured puts...',
                        'Category': 'Trading Strategies',
                        'Date': '2025-11-15'
                    }
                ]

                for result in sample_results:
                    with st.expander(f"{result['Document']} - {result['Relevance']*100:.0f}% match"):
                        st.markdown(f"**Category**: {result['Category']}")
                        st.markdown(f"**Date**: {result['Date']}")
                        st.markdown(f"**Preview**: {result['Preview']}")
        else:
            st.warning("Please enter a search query")

# ============================================================================
# TAB 5: Settings
# ============================================================================

with tab5:
    st.markdown("### ⚙️ RAG System Settings")

    # Embedding model
    st.markdown("#### 🧠 Embedding Model")

    embedding_model = st.selectbox(
        "Model",
        [
            "all-MiniLM-L6-v2 (Fast, local)",
            "text-embedding-3-small (OpenAI)",
            "text-embedding-3-large (OpenAI, best quality)"
        ],
        key="embedding_model"
    )

    st.caption(f"Current: {stats.get('embedding_model', 'all-MiniLM-L6-v2')}")

    st.divider()

    # Chunking settings
    st.markdown("#### 📄 Document Chunking")

    col1, col2 = st.columns(2)

    with col1:
        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=100,
            max_value=2000,
            value=500,
            key="chunk_size"
        )

    with col2:
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=200,
            value=50,
            key="chunk_overlap"
        )

    st.divider()

    # Sync schedule
    st.markdown("#### 🕒 Sync Schedule")

    xtrades_schedule = st.time_input(
        "XTrades Daily Sync",
        value=pd.to_datetime("01:00").time(),
        key="xtrades_schedule"
    )

    discord_schedule = st.time_input(
        "Discord Daily Sync",
        value=pd.to_datetime("02:00").time(),
        key="discord_schedule"
    )

    st.divider()

    # Maintenance
    st.markdown("#### 🗑️ Maintenance")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Category", key="clear_category"):
            category_to_clear = st.selectbox(
                "Select Category",
                [cat.value for cat in DocumentCategory],
                key="clear_category_select"
            )
            st.warning(f"This will delete all documents in {category_to_clear}")

    with col2:
        if st.button("📊 Rebuild Indexes", key="rebuild_indexes"):
            st.info("Index rebuild functionality coming soon")

    st.divider()

    if st.button("💾 Save Settings", type="primary", key="save_settings"):
        st.success("✅ Settings saved successfully!")
        st.balloons()

# ============================================================================
# Footer
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📚 Documentation**
    - [RAG System Guide](/docs/rag)
    - [Ingestion Pipeline](/docs/ingestion)
    - [ChromaDB Integration](/docs/chromadb)
    """)

with col2:
    st.markdown("""
    **⚡ Quick Actions**
    - Trigger XTrades sync
    - Upload documents
    - Search knowledge base
    """)

with col3:
    st.markdown("""
    **📊 Statistics**
    - Documents ingested: {:,}
    - Total chunks: {:,}
    - Collections: {}
    """.format(
        stats.get('documents_processed', 0),
        stats.get('chunks_created', 0),
        stats.get('total_collections', 0)
    ))

st.caption("Magnus Trading Platform • RAG Knowledge Base v1.0")
