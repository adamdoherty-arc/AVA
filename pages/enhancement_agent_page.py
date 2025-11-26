"""
Enhancement Agent Page - Meta-Feature Monitoring System
Monitors all Magnus features, tracks TODOs, and recommends improvements
"""

import streamlit as st
import os
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd


def read_markdown_file(filepath: str) -> str:
    """Read and return content of a markdown file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def extract_todos_from_file(filepath: str) -> List[Dict]:
    """Extract TODO items from a TODO.md file"""
    todos = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse TODO items (looking for checkbox patterns)
        lines = content.split('\n')
        current_sprint = None

        for line in lines:
            # Detect sprint headers
            if line.startswith('## Sprint') or line.startswith('### Sprint'):
                current_sprint = line.strip('# ').strip()
                continue

            # Detect TODO items (checkbox format)
            if '- [ ]' in line or '- [x]' in line:
                completed = '[x]' in line
                task = line.split(']', 1)[1].strip() if ']' in line else line

                todos.append({
                    'task': task,
                    'completed': completed,
                    'sprint': current_sprint or 'General',
                    'file': filepath
                })

    except Exception as e:
        pass

    return todos


def scan_feature_folder(feature_path: str) -> Dict:
    """Scan a feature folder and extract all relevant information"""
    feature_name = os.path.basename(feature_path)

    feature_info = {
        'name': feature_name,
        'path': feature_path,
        'readme': None,
        'architecture': None,
        'spec': None,
        'agent': None,
        'todos': [],
        'changelog': None,
        'wishlist': None,
        'health_score': 0,
        'status': 'Unknown'
    }

    # Read documentation files
    readme_path = os.path.join(feature_path, 'README.md')
    if os.path.exists(readme_path):
        feature_info['readme'] = read_markdown_file(readme_path)

        # Extract status from README
        if feature_info['readme']:
            if '✅' in feature_info['readme'] or 'Status**: ✅' in feature_info['readme']:
                feature_info['status'] = 'Complete'
            elif '🟡' in feature_info['readme'] or 'In Development' in feature_info['readme']:
                feature_info['status'] = 'In Development'
            elif '🔴' in feature_info['readme'] or 'Planning' in feature_info['readme']:
                feature_info['status'] = 'Planning'

    # Read other docs
    arch_path = os.path.join(feature_path, 'ARCHITECTURE.md')
    if os.path.exists(arch_path):
        feature_info['architecture'] = read_markdown_file(arch_path)

    spec_path = os.path.join(feature_path, 'SPEC.md')
    if os.path.exists(spec_path):
        feature_info['spec'] = read_markdown_file(spec_path)

    agent_path = os.path.join(feature_path, 'AGENT.md')
    if os.path.exists(agent_path):
        feature_info['agent'] = read_markdown_file(agent_path)

    # Extract TODOs
    todo_path = os.path.join(feature_path, 'TODO.md')
    if os.path.exists(todo_path):
        feature_info['todos'] = extract_todos_from_file(todo_path)

    # Read changelog
    changelog_path = os.path.join(feature_path, 'CHANGELOG.md')
    if os.path.exists(changelog_path):
        feature_info['changelog'] = read_markdown_file(changelog_path)

    # Read wishlist
    wishlist_path = os.path.join(feature_path, 'WISHLIST.md')
    if os.path.exists(wishlist_path):
        feature_info['wishlist'] = read_markdown_file(wishlist_path)

    # Calculate health score
    feature_info['health_score'] = calculate_health_score(feature_info)

    return feature_info


def calculate_health_score(feature_info: Dict) -> int:
    """Calculate health score (0-100) based on feature completeness"""
    score = 0

    # Documentation completeness (40 points)
    if feature_info['readme']:
        score += 10
    if feature_info['architecture']:
        score += 10
    if feature_info['spec']:
        score += 10
    if feature_info['agent']:
        score += 10

    # TODO completion rate (30 points)
    todos = feature_info['todos']
    if todos:
        completed = sum(1 for t in todos if t['completed'])
        completion_rate = completed / len(todos)
        score += int(30 * completion_rate)
    else:
        score += 15  # No TODOs means either complete or not started

    # Status (30 points)
    if feature_info['status'] == 'Complete':
        score += 30
    elif feature_info['status'] == 'In Development':
        score += 20
    elif feature_info['status'] == 'Planning':
        score += 10

    return min(score, 100)


def get_all_features() -> List[Dict]:
    """Scan all feature folders and return their information"""
    features_dir = 'features'
    features = []

    if os.path.exists(features_dir):
        for feature_folder in os.listdir(features_dir):
            feature_path = os.path.join(features_dir, feature_folder)
            if os.path.isdir(feature_path):
                feature_info = scan_feature_folder(feature_path)
                features.append(feature_info)

    # Sort by health score descending
    features.sort(key=lambda x: x['health_score'], reverse=True)
    return features


def show_enhancement_agent():
    """Main Enhancement Agent page"""
    st.title("🔧 Enhancement Agent")
    st.caption("Meta-Feature Monitoring & Improvement Recommender")

    # Scan all features
    with st.spinner("Scanning all features..."):
        features = get_all_features()

    if not features:
        st.warning("No features found in the features/ directory")
        st.info("Create feature folders in the features/ directory with documentation files (README.md, TODO.md, etc.)")
        return

    # Summary metrics
    st.subheader("📊 Overall System Health")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Features", len(features))

    with col2:
        avg_health = sum(f['health_score'] for f in features) / len(features)
        st.metric("Avg Health Score", f"{avg_health:.1f}%")

    with col3:
        total_todos = sum(len(f['todos']) for f in features)
        st.metric("Total TODOs", total_todos)

    with col4:
        completed_todos = sum(sum(1 for t in f['todos'] if t['completed']) for f in features)
        st.metric("Completed TODOs", completed_todos)


    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Feature Overview", "📝 All TODOs", "💡 Recommendations", "📈 Analytics"])

    with tab1:
        st.subheader("Feature Health Dashboard")

        # Create DataFrame for overview
        overview_data = []
        for feature in features:
            overview_data.append({
                'Feature': feature['name'].replace('_', ' ').title(),
                'Health Score': feature['health_score'],
                'Status': feature['status'],
                'TODOs': len(feature['todos']),
                'Completed': sum(1 for t in feature['todos'] if t['completed']),
                'Has Docs': '✅' if feature['readme'] and feature['architecture'] else '⚠️'
            })

        df_overview = pd.DataFrame(overview_data)

        # Display as interactive table
        st.dataframe(
            df_overview,
            hide_index=True,
            width='stretch',
            column_config={
                "Health Score": st.column_config.ProgressColumn(
                    "Health Score",
                    help="Feature completeness (0-100%)",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                ),
                "Feature": st.column_config.TextColumn("Feature", width="large"),
                "Status": st.column_config.TextColumn("Status", width="medium"),
                "TODOs": st.column_config.NumberColumn("TODOs"),
                "Completed": st.column_config.NumberColumn("Completed"),
                "Has Docs": st.column_config.TextColumn("Docs", width="small")
            }
        )

        # Detailed view selector
        st.subheader("📖 Feature Details")

        selected_feature = st.selectbox(
            "Select feature to view details:",
            options=[f['name'] for f in features],
            format_func=lambda x: x.replace('_', ' ').title()
        )

        if selected_feature:
            feature = next(f for f in features if f['name'] == selected_feature)

            # Feature header
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"### {feature['name'].replace('_', ' ').title()}")
            with col2:
                st.metric("Health Score", f"{feature['health_score']}%")
            with col3:
                status_emoji = {
                    'Complete': '✅',
                    'In Development': '🟡',
                    'Planning': '🔴',
                    'Unknown': '❓'
                }
                st.metric("Status", f"{status_emoji.get(feature['status'], '❓')} {feature['status']}")

            # Tabs for feature details
            feature_tabs = st.tabs(["📄 README", "🏗️ Architecture", "📋 Spec", "🤖 Agent", "✅ TODOs", "📅 Changelog", "🌟 Wishlist"])

            with feature_tabs[0]:
                if feature['readme']:
                    st.markdown(feature['readme'])
                else:
                    st.warning("No README.md found")

            with feature_tabs[1]:
                if feature['architecture']:
                    st.markdown(feature['architecture'])
                else:
                    st.warning("No ARCHITECTURE.md found")

            with feature_tabs[2]:
                if feature['spec']:
                    st.markdown(feature['spec'])
                else:
                    st.warning("No SPEC.md found")

            with feature_tabs[3]:
                if feature['agent']:
                    st.markdown(feature['agent'])
                else:
                    st.warning("No AGENT.md found")

            with feature_tabs[4]:
                if feature['todos']:
                    for todo in feature['todos']:
                        checkbox_emoji = '✅' if todo['completed'] else '⬜'
                        st.markdown(f"{checkbox_emoji} **[{todo['sprint']}]** {todo['task']}")
                else:
                    st.info("No TODOs found")

            with feature_tabs[5]:
                if feature['changelog']:
                    st.markdown(feature['changelog'])
                else:
                    st.warning("No CHANGELOG.md found")

            with feature_tabs[6]:
                if feature['wishlist']:
                    st.markdown(feature['wishlist'])
                else:
                    st.warning("No WISHLIST.md found")

    with tab2:
        st.subheader("All TODOs Across Features")

        # Collect all TODOs
        all_todos = []
        for feature in features:
            for todo in feature['todos']:
                all_todos.append({
                    'Feature': feature['name'].replace('_', ' ').title(),
                    'Sprint': todo['sprint'],
                    'Task': todo['task'],
                    'Completed': '✅' if todo['completed'] else '⬜'
                })

        if all_todos:
            df_todos = pd.DataFrame(all_todos)

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                show_completed = st.checkbox("Show completed", value=False)
            with col2:
                filter_feature = st.multiselect(
                    "Filter by feature:",
                    options=sorted(set(df_todos['Feature']))
                )

            # Apply filters
            if not show_completed:
                df_todos = df_todos[df_todos['Completed'] == '⬜']

            if filter_feature:
                df_todos = df_todos[df_todos['Feature'].isin(filter_feature)]

            st.dataframe(df_todos, hide_index=True, width='stretch')

            # Summary
            remaining = len([t for t in all_todos if t['Completed'] == '⬜'])
            completed = len([t for t in all_todos if t['Completed'] == '✅'])
            st.info(f"📊 {remaining} remaining | ✅ {completed} completed | Total: {len(all_todos)}")
        else:
            st.info("No TODOs found across all features")

    with tab3:
        st.subheader("💡 AI-Powered Recommendations")
        st.info("🚧 This feature will use LangChain agents to analyze all features and recommend improvements")

        # Placeholder recommendations
        st.markdown("### Next Steps Recommended:")

        # Analyze features and provide recommendations
        for feature in features[:5]:  # Top 5 features
            if feature['health_score'] < 70:
                with st.expander(f"⚠️ {feature['name'].replace('_', ' ').title()} needs attention ({feature['health_score']}% health)"):

                    # Provide specific recommendations
                    recommendations = []

                    if not feature['readme']:
                        recommendations.append("📄 Create README.md with feature overview")
                    if not feature['architecture']:
                        recommendations.append("🏗️ Document architecture in ARCHITECTURE.md")
                    if not feature['spec']:
                        recommendations.append("📋 Write detailed specifications in SPEC.md")
                    if not feature['todos']:
                        recommendations.append("✅ Create TODO.md with implementation roadmap")

                    pending_todos = [t for t in feature['todos'] if not t['completed']]
                    if pending_todos:
                        recommendations.append(f"✅ Complete {len(pending_todos)} pending TODOs")

                    for rec in recommendations:
                        st.markdown(f"- {rec}")

        st.markdown("### 🎯 Top Priority Actions:")

        # Calculate top priorities
        priorities = []
        for feature in features:
            if feature['health_score'] < 50:
                priorities.append(f"**{feature['name']}**: Health score below 50% - needs immediate attention")

            incomplete_todos = [t for t in feature['todos'] if not t['completed']]
            if len(incomplete_todos) > 10:
                priorities.append(f"**{feature['name']}**: {len(incomplete_todos)} incomplete TODOs - prioritize completion")

        if priorities:
            for i, priority in enumerate(priorities[:5], 1):
                st.markdown(f"{i}. {priority}")
        else:
            st.success("✅ All features are in good health!")

    with tab4:
        st.subheader("📈 System Analytics")

        # Health score distribution
        st.markdown("#### Health Score Distribution")
        health_scores = [f['health_score'] for f in features]
        st.bar_chart(pd.DataFrame({'Health Score': health_scores}, index=[f['name'] for f in features]))

        # Status distribution
        st.markdown("#### Features by Status")
        status_counts = {}
        for feature in features:
            status = feature['status']
            status_counts[status] = status_counts.get(status, 0) + 1

        st.bar_chart(pd.DataFrame(status_counts, index=['Count']))

        # TODO completion rate
        st.markdown("#### TODO Completion Rate by Feature")
        todo_data = []
        for feature in features:
            if feature['todos']:
                completed = sum(1 for t in feature['todos'] if t['completed'])
                total = len(feature['todos'])
                completion_rate = (completed / total * 100) if total > 0 else 0
                todo_data.append({
                    'Feature': feature['name'].replace('_', ' ').title(),
                    'Completion Rate': completion_rate
                })

        if todo_data:
            df_completion = pd.DataFrame(todo_data)
            st.bar_chart(df_completion.set_index('Feature'))
        else:
            st.info("No TODO data available")

    # Footer with last scan time
    st.caption(f"Last scan: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: Daily at 2:00 AM")
