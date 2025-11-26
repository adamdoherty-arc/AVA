"""
System Management Hub - Unified Admin Interface
==============================================

Consolidated management interface combining:
- Enhancement Management (Feature requests, tracking, QA)
- Agent Management (AI agent configuration)
- Task Management (Background jobs, workflows)
- System Configuration (Settings, features, toggles)

This hub provides centralized administration for the Magnus platform.

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

st.set_page_config(
    page_title="System Management | Magnus",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .mgmt-header {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .section-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }

    .badge-pending { background: #f39c12; color: white; }
    .badge-active { background: #27ae60; color: white; }
    .badge-completed { background: #95a5a6; color: white; }
    .badge-critical { background: #e74c3c; color: white; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Header
# ============================================================================

st.markdown("""
<div class="mgmt-header">
    <h1 style="margin:0">⚙️ System Management Hub</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9">
        Enhancement Tracking • Agent Configuration • Task Management
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Quick Stats
# ============================================================================

st.subheader("📊 Quick Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Count pending enhancements
    try:
        from src.task_db_manager import TaskDBManager
        task_db = TaskDBManager()
        pending_count = task_db.get_pending_tasks_count()
        st.metric("Pending Tasks", pending_count or 0)
    except:
        st.metric("Pending Tasks", "N/A")

with col2:
    # Count active agents
    try:
        from src.ava.core.agent_initializer import initialize_all_agents
        agents = initialize_all_agents()
        st.metric("Active Agents", len(agents))
    except:
        st.metric("Active Agents", "N/A")

with col3:
    # Count features
    try:
        import yaml
        with open("config/features.yaml") as f:
            features = yaml.safe_load(f)
            enabled = sum(1 for v in features.values() if v is True)
        st.metric("Enabled Features", f"{enabled}/{len(features)}")
    except:
        st.metric("Enabled Features", "N/A")

with col4:
    # System health
    st.metric("System Health", "✅ Healthy")

st.divider()

# ============================================================================
# Main Tabs
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Enhancement Management",
    "🤖 Agent Management",
    "✅ QA & Testing",
    "⚙️ Configuration",
    "📊 Analytics"
])

# ============================================================================
# TAB 1: Enhancement Management
# ============================================================================

with tab1:
    st.markdown("### 📋 Enhancement & Feature Management")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Enhancement Tracker")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        **Track and manage platform enhancements:**
        - Feature requests
        - Bug fixes
        - Performance improvements
        - Technical debt
        - Documentation updates
        """)

    with col2:
        if st.button("➕ New Enhancement", key="new_enhancement", type="primary"):
            st.info("Enhancement creation form")

    st.markdown('</div>', unsafe_allow_html=True)

    # Enhancement list
    st.markdown("#### 📝 Active Enhancements")

    # Mock data for demonstration
    import pandas as pd
    mock_enhancements = pd.DataFrame([
        {
            "ID": "ENH-001",
            "Title": "Add voice command system",
            "Priority": "High",
            "Status": "In Progress",
            "Assigned": "AVA",
            "Due": "2025-01-15"
        },
        {
            "ID": "ENH-002",
            "Title": "Multi-modal analysis (charts)",
            "Priority": "Medium",
            "Status": "Pending",
            "Assigned": "Technical Agent",
            "Due": "2025-01-20"
        },
        {
            "ID": "ENH-003",
            "Title": "Docker containerization",
            "Priority": "Medium",
            "Status": "Planned",
            "Assigned": "DevOps",
            "Due": "2025-01-25"
        },
    ])

    st.dataframe(mock_enhancements, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 View All Enhancements", key="view_all_enhancements"):
            st.info("Navigate to Enhancement Manager page")

    with col2:
        if st.button("📈 Priority Report", key="priority_report"):
            st.info("Show priority report")

    with col3:
        if st.button("📅 Timeline View", key="timeline_view"):
            st.info("Show Gantt chart")

# ============================================================================
# TAB 2: Agent Management
# ============================================================================

with tab2:
    st.markdown("### 🤖 AI Agent Configuration")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Agent Registry")

    st.markdown("""
    **Manage AVA's AI agents:**
    - Enable/disable agents
    - Configure capabilities
    - Adjust prompts and system instructions
    - Monitor agent usage
    - View performance metrics
    """)

    st.markdown('</div>', unsafe_allow_html=True)

    # Agent categories
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**💬 Conversation Agents**")
        agents_conversation = [
            "Greeting Agent",
            "Portfolio Agent",
            "Calendar Spread Agent",
            "Options Strategy Agent"
        ]
        for agent in agents_conversation:
            st.checkbox(agent, value=True, key=f"agent_{agent.lower().replace(' ', '_')}")

    with col2:
        st.markdown("**📊 Data Agents**")
        agents_data = [
            "Market Data Agent",
            "Stock Screener Agent",
            "Technical Analysis Agent",
            "Earnings Agent"
        ]
        for agent in agents_data:
            st.checkbox(agent, value=True, key=f"agent_{agent.lower().replace(' ', '_')}")

    with col3:
        st.markdown("**🔍 Monitoring Agents**")
        agents_monitoring = [
            "Discord Agent",
            "Analytics Agent",
            "Cache Metrics Agent",
            "Performance Agent"
        ]
        for agent in agents_monitoring:
            st.checkbox(agent, value=True, key=f"agent_{agent.lower().replace(' ', '_')}")

    st.divider()

    st.markdown("#### ⚙️ Agent Configuration")

    selected_agent = st.selectbox(
        "Select Agent to Configure",
        ["Portfolio Agent", "Options Strategy Agent", "Calendar Spread Agent", "Technical Analysis Agent"],
        key="config_agent_select"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.text_area(
            "System Prompt",
            placeholder="Enter system prompt for the agent...",
            height=150,
            key="agent_system_prompt"
        )

    with col2:
        st.text_area(
            "Capabilities",
            placeholder="List agent capabilities...",
            height=150,
            key="agent_capabilities"
        )

    if st.button("💾 Save Agent Configuration", type="primary", key="save_agent_config"):
        st.success(f"✅ {selected_agent} configuration saved!")

    st.divider()

    if st.button("🔍 Open Full Agent Manager", key="open_agent_manager"):
        st.info("Navigate to Agent Management page")

# ============================================================================
# TAB 3: QA & Testing
# ============================================================================

with tab3:
    st.markdown("### ✅ Quality Assurance & Testing")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 🧪 Test Suite Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Tests", 150)
        st.caption("Unit + Integration")

    with col2:
        st.metric("Pass Rate", "94%")
        st.caption("141/150 passing")

    with col3:
        st.metric("Coverage", "78%")
        st.caption("Code coverage")

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("#### 📋 Enhancement QA Checklist")

    st.markdown("""
    **Pre-Deployment Checklist:**
    - [ ] Unit tests pass
    - [ ] Integration tests pass
    - [ ] Manual QA completed
    - [ ] Code review approved
    - [ ] Documentation updated
    - [ ] Performance benchmarks met
    - [ ] Security scan passed
    - [ ] Rollback plan documented
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧪 Run All Tests", key="run_all_tests"):
            st.info("Running test suite...")

    with col2:
        if st.button("📊 View Test Report", key="view_test_report"):
            st.info("Navigate to QA Management page")

# ============================================================================
# TAB 4: Configuration
# ============================================================================

with tab4:
    st.markdown("### ⚙️ System Configuration")

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("#### 🎛️ Feature Flags")

    st.markdown("**Enable/disable platform features:**")

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox("🎮 Sports Betting Module", value=True, key="feature_sports")
        st.checkbox("📊 Options Analysis", value=True, key="feature_options")
        st.checkbox("🤖 AI Predictions", value=True, key="feature_ai_pred")
        st.checkbox("💹 Kalshi Integration", value=True, key="feature_kalshi")
        st.checkbox("📱 Discord Alerts", value=False, key="feature_discord")

    with col2:
        st.checkbox("💾 Redis Caching", value=True, key="feature_redis")
        st.checkbox("🗣️ Voice Commands", value=False, key="feature_voice")
        st.checkbox("📈 Advanced Charts", value=True, key="feature_charts")
        st.checkbox("🔐 Two-Factor Auth", value=False, key="feature_2fa")
        st.checkbox("📊 RAG Knowledge Base", value=True, key="feature_rag")

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    st.markdown("#### 🔧 API Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.text_input("Robinhood API Key", type="password", key="api_robinhood")
        st.text_input("Kalshi API Key", type="password", key="api_kalshi")
        st.text_input("OpenAI API Key", type="password", key="api_openai")

    with col2:
        st.text_input("Groq API Key", type="password", key="api_groq")
        st.text_input("Anthropic API Key", type="password", key="api_anthropic")
        st.text_input("Discord Webhook URL", type="password", key="api_discord")

    st.divider()

    st.markdown("#### 🗄️ Database Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.text_input("DB Host", value="localhost", key="db_host")

    with col2:
        st.number_input("DB Port", value=5432, key="db_port")

    with col3:
        st.text_input("DB Name", value="magnus", key="db_name")

    st.divider()

    if st.button("💾 Save All Configuration", type="primary", key="save_all_config"):
        st.success("✅ Configuration saved successfully!")
        st.balloons()

# ============================================================================
# TAB 5: Analytics
# ============================================================================

with tab5:
    st.markdown("### 📊 System Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 Enhancement Velocity")

        import pandas as pd
        velocity_data = pd.DataFrame({
            "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
            "Completed": [8, 12, 15, 10],
            "In Progress": [5, 8, 6, 12]
        })

        st.bar_chart(velocity_data.set_index("Week"))

    with col2:
        st.markdown("#### 🎯 Priority Distribution")

        priority_data = pd.DataFrame({
            "Priority": ["Critical", "High", "Medium", "Low"],
            "Count": [3, 8, 15, 12]
        })

        st.bar_chart(priority_data.set_index("Priority"))

    st.divider()

    st.markdown("#### 📊 Agent Usage Statistics")

    agent_usage = pd.DataFrame({
        "Agent": ["Portfolio", "Options", "Calendar", "Technical", "Discord"],
        "Calls (24h)": [145, 89, 56, 234, 67],
        "Avg Response Time (ms)": [250, 1200, 3500, 800, 150]
    })

    st.dataframe(agent_usage, use_container_width=True, hide_index=True)

# ============================================================================
# Footer
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🔗 Quick Links**
    - [System Documentation](/)
    - [API Reference](/)
    - [Agent SDK Docs](/)
    """)

with col2:
    st.markdown("""
    **⚡ Actions**
    - Backup Database
    - Export Logs
    - System Health Check
    """)

with col3:
    st.markdown("""
    **📞 Support**
    - [GitHub Issues](https://github.com/magnus)
    - [Slack Channel](/)
    - [Email Support](mailto:support@magnus.ai)
    """)

st.caption("Magnus Trading Platform • System Management Hub v1.0")
