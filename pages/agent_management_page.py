"""
Agent Management Dashboard
Display and manage all agents, their status, performance, and activities
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import logging
import sys
import traceback

# Fix Windows encoding for emoji characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass  # Silently fail if already configured

# Import agent system
from src.ava.core.agent_registry import AgentRegistry
from src.ava.core.agent_learning import AgentLearningSystem
from src.ava.core.agent_base import BaseAgent
from src.ava.core.agent_initializer import ensure_agents_initialized, get_registry

logger = logging.getLogger(__name__)

@st.cache_resource
def get_initialized_registry():
    """Get or create initialized registry with agents - cached across reruns"""
    print("=" * 60)
    print("🔍 DEBUG: get_initialized_registry() called")
    print("=" * 60)

    ensure_agents_initialized()
    print("🔍 DEBUG: ensure_agents_initialized() completed")

    registry = get_registry()
    print(f"🔍 DEBUG: get_registry() returned: {registry}")
    print(f"🔍 DEBUG: Registry type: {type(registry)}")

    all_agents = registry.get_all_agents()
    agent_count = len(all_agents)
    print(f"🔍 DEBUG: Registry.get_all_agents() returned {agent_count} agents")

    if agent_count == 0:
        print("🔍 DEBUG: WARNING - No agents in registry!")
        print(f"🔍 DEBUG: Registry._agents dict: {registry._agents}")
        print(f"🔍 DEBUG: Registry.list_agent_names(): {registry.list_agent_names()}")
    else:
        print(f"🔍 DEBUG: First 3 agents:")
        for i, agent in enumerate(all_agents[:3]):
            if agent:
                print(f"  {i+1}. {agent.name}")

    print("=" * 60)
    return registry

def main():
    """Main function for agent management page"""
    st.title("🤖 Agent Management Dashboard")
    st.markdown("Monitor and manage all AVA agents, their performance, and activities")

    # Get cached registry with agents
    try:
        print("\n🔍 DEBUG: main() - About to call get_initialized_registry()")
        registry = get_initialized_registry()
        print(f"🔍 DEBUG: main() - Registry received")

        all_agents = registry.get_all_agents()
        agent_count = len(all_agents)
        print(f"🔍 DEBUG: main() - Got {agent_count} agents\n")

        if agent_count > 0:
            st.success(f"✅ {agent_count} agents loaded successfully")
            # Debug display
            with st.expander("🔍 Debug Info (Expand to see details)"):
                st.write(f"**Registry type:** `{type(registry)}`")
                st.write(f"**Agent count:** {agent_count}")
                st.write(f"**Agent names:**")
                st.code("\n".join(registry.list_agent_names()))
        else:
            st.error(f"❌ No agents found in registry!")
            st.warning("🔍 **Check your terminal/console** for debug output")

            # Show debug info
            with st.expander("🔍 Debug Info - IMPORTANT"):
                st.code(f"""
Registry object: {registry}
Registry type: {type(registry)}
Registry._agents: {getattr(registry, '_agents', 'N/A')}
list_agent_names(): {registry.list_agent_names()}
get_all_agents(): {all_agents}
                """)

            # Retry button
            if st.button("🔄 Clear Cache & Retry"):
                st.cache_resource.clear()
                st.rerun()
            return
    except Exception as e:
        st.error(f"❌ Error initializing agents: {e}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return
    
    # Get learning system
    try:
        learning_system = AgentLearningSystem()
    except Exception as e:
        st.warning(f"Could not initialize learning system: {e}")
        learning_system = None
    
    # Get performance data
    try:
        if learning_system:
            all_performance = learning_system.get_all_performance()
        else:
            all_performance = []
    except Exception as e:
        st.warning(f"Could not get performance data: {e}")
        all_performance = []
    
    # Sidebar filters
    st.sidebar.header("Filters")
    filter_category = st.sidebar.selectbox(
        "Category",
        ["All", "Trading", "Sports Betting", "Analysis", "Monitoring", "Research", "Management", "Code Development"]
    )
    
    filter_status = st.sidebar.selectbox(
        "Status",
        ["All", "Active", "Inactive", "Error"]
    )
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🤖 Agents",
        "🚀 Execute Agent",
        "📈 Performance",
        "💾 Memory & Learning",
        "📝 Activity Log"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("System Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Agents", agent_count)
        
        with col2:
            total_executions = sum(p.total_executions for p in all_performance) if all_performance else 0
            st.metric("Total Executions", f"{total_executions:,}")
        
        with col3:
            avg_success_rate = sum(p.success_rate for p in all_performance) / len(all_performance) if all_performance else 0
            st.metric("Avg Success Rate", f"{avg_success_rate:.1%}")
        
        with col4:
            active_agents = len([a for a in all_agents if a and hasattr(a, 'name')])
            st.metric("Active Agents", active_agents)
        
        # Agent categories
        st.subheader("Agents by Category")
        
        categories = {
            "Trading": ["market_data", "options", "strategy", "risk", "portfolio", "earnings", "premium", "watchlist"],
            "Sports Betting": ["kalshi", "sports", "nfl", "game", "odds", "betting"],
            "Analysis": ["fundamental", "technical", "sentiment", "supply", "sector", "options_flow"],
            "Monitoring": ["watchlist_monitor", "xtrades", "alert", "price_action"],
            "Research": ["knowledge", "research", "documentation"],
            "Management": ["task", "position", "settings", "integration"],
            "Code Development": ["code_recommendation", "claude_code", "qa"]
        }
        
        category_counts = {}
        for category, keywords in categories.items():
            count = sum(1 for agent in all_agents if agent and any(kw in agent.name.lower() for kw in keywords))
            category_counts[category] = count
        
        if category_counts:
            fig = px.bar(
                x=list(category_counts.keys()),
                y=list(category_counts.values()),
                title="Agents by Category",
                labels={"x": "Category", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No agents to categorize")
    
    # Tab 2: Agents
    with tab2:
        st.header("All Agents")
        
        if not all_agents or len(all_agents) == 0:
            st.error("❌ No agents found!")
            return
        
        # Agent list
        agents_data = []
        for agent in all_agents:
            if not agent:
                continue
            try:
                perf = learning_system.get_performance(agent.name) if learning_system else None
                capabilities = agent.metadata.get('capabilities', []) if hasattr(agent, 'metadata') else []
                agents_data.append({
                    "Name": agent.name,
                    "Description": agent.description,
                    "Category": next((cat for cat, keywords in categories.items() if any(kw in agent.name.lower() for kw in keywords)), "Other"),
                    "Capabilities": len(capabilities),
                    "Tools": len(agent.tools) if hasattr(agent, 'tools') else 0,
                    "Total Executions": perf.total_executions if perf else 0,
                    "Success Rate": f"{perf.success_rate:.1%}" if perf else "N/A",
                    "Last Execution": perf.last_execution.strftime("%Y-%m-%d %H:%M") if perf and perf.last_execution else "Never"
                })
            except Exception as e:
                logger.warning(f"Error processing agent {agent.name if agent else 'unknown'}: {e}")
                continue
        
        if agents_data:
            df_agents = pd.DataFrame(agents_data)
            
            # Filters
            if filter_category != "All":
                df_agents = df_agents[df_agents["Category"] == filter_category]
            
            st.dataframe(df_agents, use_container_width=True, hide_index=True)
            
            # Agent details
            st.subheader("Agent Details")
            if all_agents:
                selected_agent_name = st.selectbox(
                    "Select Agent",
                    [agent.name for agent in all_agents if agent]
                )
                
                if selected_agent_name:
                    agent = registry.get_agent(selected_agent_name)
                    perf = learning_system.get_performance(selected_agent_name) if learning_system else None
                    
                    if agent:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Agent Information**")
                            st.write(f"**Name:** {agent.name}")
                            st.write(f"**Description:** {agent.description}")
                            capabilities = agent.metadata.get('capabilities', []) if hasattr(agent, 'metadata') else []
                            st.write(f"**Capabilities:** {', '.join(capabilities) if capabilities else 'None'}")
                            st.write(f"**Tools:** {len(agent.tools) if hasattr(agent, 'tools') else 0}")
                            if hasattr(agent, 'use_huggingface'):
                                st.write(f"**Hugging Face:** {'✅ Enabled' if agent.use_huggingface else '❌ Disabled'}")
                        
                        with col2:
                            st.write("**Performance Metrics**")
                            if perf:
                                st.write(f"**Total Executions:** {perf.total_executions}")
                                st.write(f"**Success Rate:** {perf.success_rate:.1%}")
                                st.write(f"**Avg Response Time:** {perf.average_response_time:.2f}ms")
                                st.write(f"**Last Execution:** {perf.last_execution}")
                            else:
                                st.write("No performance data available")
                                st.info("This agent hasn't been executed yet. Performance data will appear after first execution.")
        else:
            st.warning("No agent data available")

    # Tab 3: Execute Agent
    with tab3:
        st.header("🚀 Execute Agent")
        st.markdown("Test and execute agents interactively")

        # Agent selection
        col1, col2 = st.columns([2, 1])

        with col1:
            exec_agent_name = st.selectbox(
                "Select Agent to Execute",
                [agent.name for agent in all_agents if agent],
                key="exec_agent"
            )

        with col2:
            st.metric("Agent Tools", len(registry.get_agent(exec_agent_name).tools) if exec_agent_name and registry.get_agent(exec_agent_name) else 0)

        if exec_agent_name:
            exec_agent = registry.get_agent(exec_agent_name)

            if exec_agent:
                # Agent info
                with st.expander("ℹ️ Agent Information", expanded=False):
                    st.write(f"**Name:** {exec_agent.name}")
                    st.write(f"**Description:** {exec_agent.description}")
                    capabilities = exec_agent.metadata.get('capabilities', []) if hasattr(exec_agent, 'metadata') else []
                    if capabilities:
                        st.write(f"**Capabilities:** {', '.join(capabilities)}")

                # Input area
                st.subheader("Input")

                # Quick templates
                templates = {
                    "Market Data Agent": "Get current market data for TSLA",
                    "Options Analysis Agent": "Analyze options chain for AAPL with 30-45 DTE",
                    "Sports Betting Agent": "Analyze upcoming NFL games for betting opportunities",
                    "Kalshi Markets Agent": "Get current Kalshi markets for NFL games",
                    "Game Analysis Agent": "Analyze Dallas Cowboys vs Las Vegas Raiders",
                    "Technical Analysis Agent": "Perform technical analysis on SPY",
                    "Risk Management Agent": "Analyze portfolio risk for current positions",
                    "General": "What can you help me with?"
                }

                # Template selector
                template_key = st.selectbox(
                    "Quick Templates (optional)",
                    ["Custom"] + list(templates.keys()),
                    key="template"
                )

                if template_key != "Custom" and template_key in templates:
                    default_input = templates[template_key]
                else:
                    default_input = ""

                user_input = st.text_area(
                    "Enter your input/question for the agent:",
                    value=default_input,
                    height=100,
                    placeholder="e.g., 'Analyze TSLA options for wheel strategy'"
                )

                # Execution context
                with st.expander("⚙️ Advanced Options", expanded=False):
                    use_context = st.checkbox("Include system context", value=True)
                    use_tools = st.checkbox("Enable agent tools", value=True)
                    stream_response = st.checkbox("Stream response", value=False)

                # Execute button
                col_exec, col_clear = st.columns([3, 1])

                with col_exec:
                    execute_btn = st.button("🚀 Execute Agent", type="primary", use_container_width=True)

                with col_clear:
                    if st.button("🗑️ Clear", use_container_width=True):
                        st.rerun()

                # Execute the agent
                if execute_btn and user_input:
                    st.subheader("Response")

                    with st.spinner(f"Executing {exec_agent.name}..."):
                        try:
                            from datetime import datetime
                            start_time = datetime.now()

                            # Prepare context
                            context = {}
                            if use_context:
                                context = {
                                    "platform": "agent_management_dashboard",
                                    "user_type": "admin",
                                    "timestamp": datetime.now().isoformat()
                                }

                            # Execute agent
                            response = exec_agent.execute(
                                input=user_input,
                                context=context
                            )

                            end_time = datetime.now()
                            execution_time = (end_time - start_time).total_seconds() * 1000

                            # Display response
                            st.success("✅ Execution Complete")

                            # Metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Execution Time", f"{execution_time:.0f}ms")
                            with col2:
                                st.metric("Status", "Success")
                            with col3:
                                st.metric("Response Length", f"{len(str(response))} chars")

                            # Response content
                            st.markdown("### Agent Response:")

                            if isinstance(response, dict):
                                if 'output' in response:
                                    st.markdown(response['output'])
                                elif 'result' in response:
                                    st.markdown(response['result'])
                                elif 'response' in response:
                                    st.markdown(response['response'])
                                else:
                                    st.json(response)
                            else:
                                st.markdown(str(response))

                            # Show raw response in expander
                            with st.expander("🔍 Raw Response Data"):
                                st.json(response if isinstance(response, dict) else {"response": str(response)})

                            # Log execution
                            if learning_system:
                                try:
                                    learning_system.log_execution(
                                        agent_name=exec_agent.name,
                                        execution_id=f"manual_{int(start_time.timestamp())}",
                                        input_text=user_input,
                                        output_text=str(response)[:1000],
                                        response_time_ms=execution_time,
                                        success=True,
                                        platform="agent_management_dashboard"
                                    )
                                except Exception as log_error:
                                    st.caption(f"Note: Execution logged with warning: {log_error}")

                        except Exception as e:
                            st.error(f"❌ Execution Failed: {e}")

                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())

                            # Log failed execution
                            if learning_system:
                                try:
                                    learning_system.log_execution(
                                        agent_name=exec_agent.name,
                                        execution_id=f"manual_{int(datetime.now().timestamp())}",
                                        input_text=user_input,
                                        output_text="",
                                        response_time_ms=0,
                                        success=False,
                                        error=str(e),
                                        platform="agent_management_dashboard"
                                    )
                                except:
                                    pass

                elif execute_btn and not user_input:
                    st.warning("⚠️ Please enter an input/question for the agent")

                # Help section
                with st.expander("💡 Tips & Examples"):
                    st.markdown("""
                    **How to use:**
                    1. Select an agent from the dropdown
                    2. Choose a template or enter custom input
                    3. Click "Execute Agent" to run

                    **Example Inputs:**
                    - **Market Data**: "Get current price and volume for NVDA"
                    - **Options Analysis**: "Find best CSP opportunities for MSFT"
                    - **Sports Betting**: "Analyze NFL games this weekend"
                    - **Technical Analysis**: "Show support and resistance for SPY"
                    - **Risk Management**: "Calculate position sizing for new trade"

                    **Note:** Agent responses depend on available tools and data sources.
                    """)

    # Tab 4: Performance
    with tab4:
        st.header("Performance Metrics")
        
        # Performance chart
        if all_performance:
            perf_data = {
                "Agent": [p.agent_name for p in all_performance],
                "Success Rate": [p.success_rate for p in all_performance],
                "Total Executions": [p.total_executions for p in all_performance],
                "Avg Response Time (ms)": [p.average_response_time for p in all_performance]
            }
            
            df_perf = pd.DataFrame(perf_data)
            
            # Success rate chart
            fig = px.bar(
                df_perf,
                x="Agent",
                y="Success Rate",
                title="Success Rate by Agent",
                labels={"Success Rate": "Success Rate (%)"}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Response time chart
            fig2 = px.bar(
                df_perf,
                x="Agent",
                y="Avg Response Time (ms)",
                title="Average Response Time by Agent"
            )
            fig2.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Performance table
            st.subheader("Performance Table")
            st.dataframe(df_perf, use_container_width=True, hide_index=True)
        else:
            st.info("No performance data available yet. Performance metrics will appear after agents are executed.")
    
    # Tab 5: Memory & Learning
    with tab5:
        st.header("Memory & Learning")
        
        st.subheader("Agent Memory")
        selected_agent_memory = st.selectbox(
            "Select Agent for Memory",
            [agent.name for agent in all_agents if agent],
            key="memory_agent"
        )
        
        if selected_agent_memory:
            st.write(f"Memory for {selected_agent_memory}")
            st.info("Memory viewing feature - to be implemented with database queries")
        
        st.subheader("Learning Statistics")
        st.write("**Feedback Collection**")
        try:
            import psycopg2
            import os
            from psycopg2.extras import RealDictCursor
            
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'magnus'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }
            
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT feedback_type, COUNT(*) as count 
                FROM agent_feedback 
                GROUP BY feedback_type
            """)
            feedback_stats = cur.fetchall()
            cur.close()
            conn.close()
            
            if feedback_stats:
                for stat in feedback_stats:
                    st.write(f"- **{stat['feedback_type']}**: {stat['count']} feedback entries")
            else:
                st.info("No feedback collected yet.")
        except Exception as e:
            st.warning(f"Could not fetch feedback: {e}")
        
        st.write("**Auto-Update Status**")
        st.success("✅ Learning system active - All agent executions are automatically logged and tracked")
    
    # Tab 6: Activity Log
    with tab6:
        st.header("Recent Activity")
        
        # Activity log
        st.write("**Recent Executions**")
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            agent_options = ["All"] + [agent.name for agent in all_agents if agent]
            log_agent_filter = st.selectbox(
                "Filter by Agent",
                agent_options,
                key="log_agent"
            )
        with col2:
            log_time_filter = st.selectbox(
                "Time Range",
                ["Last Hour", "Last 24 Hours", "Last Week", "All Time"],
                key="log_time"
            )
        
        # Query execution log
        try:
            import psycopg2
            import os
            from datetime import datetime, timedelta
            from psycopg2.extras import RealDictCursor
            
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'magnus'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }
            
            # Calculate time filter
            time_filters = {
                "Last Hour": timedelta(hours=1),
                "Last 24 Hours": timedelta(days=1),
                "Last Week": timedelta(weeks=1),
                "All Time": None
            }
            time_delta = time_filters.get(log_time_filter)
            
            conn = psycopg2.connect(**db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT agent_name, execution_id, input_text, 
                       CASE WHEN error IS NULL THEN 'Success' ELSE 'Error' END as status,
                       response_time_ms, timestamp, platform
                FROM agent_execution_log
                WHERE 1=1
            """
            params = []
            
            if log_agent_filter != "All":
                query += " AND agent_name = %s"
                params.append(log_agent_filter)
            
            if time_delta:
                query += " AND timestamp >= %s"
                params.append(datetime.now() - time_delta)
            
            query += " ORDER BY timestamp DESC LIMIT 100"
            
            cur.execute(query, params)
            executions = cur.fetchall()
            cur.close()
            conn.close()
            
            if executions:
                st.write(f"**Showing {len(executions)} recent executions:**")
                for exec_log in executions[:20]:  # Show top 20
                    status_emoji = "✅" if exec_log['status'] == 'Success' else "❌"
                    st.write(f"{status_emoji} **{exec_log['agent_name']}** - {exec_log['timestamp']} ({exec_log['response_time_ms']:.0f}ms) - {exec_log['platform']}")
                    if exec_log['input_text']:
                        st.caption(f"Input: {exec_log['input_text'][:100]}...")
            else:
                st.info("No executions found for selected filters.")
        except Exception as e:
            st.warning(f"Could not fetch execution log: {e}")
            st.info("Execution log requires database connection.")
    
    # Footer
    st.markdown("**Agent Management Dashboard** - Monitor and manage all AVA agents")

if __name__ == "__main__":
    main()
