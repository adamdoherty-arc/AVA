"""
Enhancement & QA Management Page
=================================

Comprehensive view of all enhancements, features, and their QA approval status.
Shows which agents signed off on each task for full transparency.

Features:
- All tasks with detailed QA sign-off information
- Agent-specific approval status
- Legion integration status
- Timeline view of approvals
- Filterable by status, agent, priority
"""

import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()


class EnhancementQAManager:
    """Enhanced task management with prominent QA sign-off display"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'magnus'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }

    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def get_all_tasks_with_qa_details(self, limit=100):
        """Get all tasks with complete QA sign-off details"""
        query = """
            WITH task_sign_offs AS (
                SELECT
                    task_id,
                    COUNT(*) as total_sign_offs,
                    COUNT(*) FILTER (WHERE status = 'approved') as approved_count,
                    COUNT(*) FILTER (WHERE status = 'rejected') as rejected_count,
                    COUNT(*) FILTER (WHERE status = 'pending') as pending_count,
                    jsonb_agg(
                        jsonb_build_object(
                            'agent', agent_name,
                            'status', status,
                            'reviewed_at', reviewed_at,
                            'notes', notes
                        ) ORDER BY created_at
                    ) as sign_off_details
                FROM qa_agent_sign_offs
                GROUP BY task_id
            ),
            task_qa_issues AS (
                SELECT
                    parent_task_id,
                    COUNT(*) FILTER (WHERE status IN ('open', 'in_progress')) as open_issues,
                    COUNT(*) FILTER (WHERE status = 'resolved') as resolved_issues,
                    jsonb_agg(
                        jsonb_build_object(
                            'id', id,
                            'title', title,
                            'severity', severity,
                            'status', status,
                            'reported_by', reported_by_agent,
                            'created_at', created_at
                        )
                    ) FILTER (WHERE status IN ('open', 'in_progress')) as open_issue_details
                FROM qa_tasks
                GROUP BY parent_task_id
            )
            SELECT
                t.id,
                t.title,
                t.description,
                t.task_type,
                t.priority,
                t.status,
                t.assigned_agent,
                t.feature_area,
                t.created_at,
                t.updated_at,
                t.completed_at,
                t.estimated_duration_minutes,
                t.tags,
                t.metadata,
                COALESCE(tso.total_sign_offs, 0) as total_sign_offs_required,
                COALESCE(tso.approved_count, 0) as sign_offs_approved,
                COALESCE(tso.rejected_count, 0) as sign_offs_rejected,
                COALESCE(tso.pending_count, 0) as sign_offs_pending,
                COALESCE(tso.sign_off_details, '[]'::jsonb) as sign_off_details,
                COALESCE(tqi.open_issues, 0) as open_qa_issues,
                COALESCE(tqi.resolved_issues, 0) as resolved_qa_issues,
                COALESCE(tqi.open_issue_details, '[]'::jsonb) as open_issue_details,
                CASE
                    WHEN t.status = 'qa_approved' THEN true
                    WHEN tso.total_sign_offs > 0 AND
                         tso.approved_count = tso.total_sign_offs AND
                         COALESCE(tqi.open_issues, 0) = 0 THEN true
                    ELSE false
                END as ready_to_finalize
            FROM development_tasks t
            LEFT JOIN task_sign_offs tso ON t.id = tso.task_id
            LEFT JOIN task_qa_issues tqi ON t.id = tqi.parent_task_id
            ORDER BY
                CASE priority
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                    ELSE 5
                END,
                t.created_at DESC
            LIMIT %s;
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, (limit,))
                return [dict(row) for row in cur.fetchall()]

    def get_qa_summary_stats(self):
        """Get summary statistics for QA dashboard"""
        query = """
            SELECT
                COUNT(*) as total_tasks,
                COUNT(*) FILTER (WHERE status = 'qa_approved') as qa_approved,
                COUNT(*) FILTER (WHERE status = 'completed') as awaiting_qa,
                COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress,
                COUNT(*) FILTER (WHERE status = 'pending') as pending,
                (
                    SELECT COUNT(*) FROM qa_agent_sign_offs
                    WHERE status = 'pending'
                ) as pending_sign_offs,
                (
                    SELECT COUNT(*) FROM qa_tasks
                    WHERE status IN ('open', 'in_progress')
                ) as open_qa_issues
            FROM development_tasks;
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                return dict(cur.fetchone())


def render_sign_off_badge(sign_off):
    """Render a visual badge for agent sign-off status"""
    if sign_off['status'] == 'approved':
        return f"✅ **{sign_off['agent']}** (Approved)"
    elif sign_off['status'] == 'rejected':
        return f"❌ **{sign_off['agent']}** (Rejected)"
    else:
        return f"⏳ **{sign_off['agent']}** (Pending)"


def render_task_card(task):
    """Render an enhanced task card with prominent QA information"""
    # Determine overall status color and emoji
    if task['status'] == 'qa_approved':
        status_color = "green"
        status_emoji = "✅"
        status_text = "QA APPROVED"
    elif task['ready_to_finalize']:
        status_color = "blue"
        status_emoji = "🎯"
        status_text = "READY TO FINALIZE"
    elif task['sign_offs_rejected'] > 0 or task['open_qa_issues'] > 0:
        status_color = "red"
        status_emoji = "🔴"
        status_text = "ISSUES FOUND"
    elif task['status'] == 'completed':
        status_color = "orange"
        status_emoji = "⏳"
        status_text = "AWAITING QA"
    elif task['status'] == 'in_progress':
        status_color = "blue"
        status_emoji = "🔵"
        status_text = "IN PROGRESS"
    else:
        status_color = "gray"
        status_emoji = "⚪"
        status_text = "PENDING"

    # Priority badge
    priority_colors = {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢'
    }
    priority_emoji = priority_colors.get(task['priority'], '⚪')

    with st.container():
        # Header with status
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            st.markdown(f"### Task #{task['id']}")

        with col2:
            st.markdown(f"#### {task['title']}")

        with col3:
            st.markdown(f"**{status_emoji} {status_text}**")

        # Quick info bar
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)

        with info_col1:
            st.caption(f"Priority: {priority_emoji} {task['priority'].upper()}")

        with info_col2:
            st.caption(f"Type: {task['task_type']}")

        with info_col3:
            st.caption(f"Agent: {task['assigned_agent'] or 'Unassigned'}")

        with info_col4:
            if task['feature_area']:
                st.caption(f"Area: {task['feature_area']}")

        st.divider()

        # QA Sign-off Section - PROMINENT
        if task['total_sign_offs_required'] > 0:
            st.markdown("### 🔍 QA Sign-Offs")

            # Progress bar
            approval_progress = task['sign_offs_approved'] / task['total_sign_offs_required'] if task['total_sign_offs_required'] > 0 else 0
            st.progress(approval_progress, text=f"{task['sign_offs_approved']}/{task['total_sign_offs_required']} Approved")

            # Individual sign-offs
            sign_off_cols = st.columns(min(task['total_sign_offs_required'], 4))
            for idx, sign_off in enumerate(task['sign_off_details']):
                col_idx = idx % 4
                with sign_off_cols[col_idx]:
                    badge_text = render_sign_off_badge(sign_off)
                    st.markdown(badge_text)

                    if sign_off['reviewed_at']:
                        review_date = datetime.fromisoformat(str(sign_off['reviewed_at']))
                        st.caption(f"Reviewed: {review_date.strftime('%Y-%m-%d %H:%M')}")

                    if sign_off.get('notes'):
                        with st.expander("View Notes"):
                            st.write(sign_off['notes'])

        # QA Issues Section
        if task['open_qa_issues'] > 0:
            st.markdown("### 🔴 Open QA Issues")
            st.error(f"{task['open_qa_issues']} issue(s) must be resolved before finalization")

            for issue in task['open_issue_details']:
                with st.expander(f"Issue: {issue['title']}"):
                    st.write(f"**Severity:** {issue['severity']}")
                    st.write(f"**Reported by:** {issue['reported_by']}")
                    st.write(f"**Status:** {issue['status']}")

                    if issue.get('created_at'):
                        issue_date = datetime.fromisoformat(str(issue['created_at']))
                        st.write(f"**Created:** {issue_date.strftime('%Y-%m-%d %H:%M')}")

        # Task Details
        with st.expander("📋 Task Details"):
            if task['description']:
                st.markdown("**Description:**")
                st.write(task['description'][:500] + "..." if len(task['description']) > 500 else task['description'])

            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.write(f"**Created:** {task['created_at'].strftime('%Y-%m-%d') if task['created_at'] else 'N/A'}")
                if task['completed_at']:
                    st.write(f"**Completed:** {task['completed_at'].strftime('%Y-%m-%d')}")
                if task['estimated_duration_minutes']:
                    st.write(f"**Estimated Duration:** {task['estimated_duration_minutes']} min")

            with detail_col2:
                if task['tags']:
                    st.write("**Tags:**")
                    st.write(", ".join(task['tags']))
                if task.get('metadata') and task['metadata'].get('legion_task_id'):
                    st.write(f"**Legion ID:** {task['metadata']['legion_task_id']}")

        st.divider()


def main():
    st.set_page_config(
        page_title="Enhancement & QA Management",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 Enhancement & QA Management")
    st.caption("Comprehensive view of all enhancements with QA sign-off tracking")

    manager = EnhancementQAManager()

    # Get summary stats
    stats = manager.get_qa_summary_stats()

    # Display summary metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Tasks", stats['total_tasks'])

    with col2:
        st.metric("QA Approved", stats['qa_approved'], delta=None, delta_color="normal")

    with col3:
        st.metric("Awaiting QA", stats['awaiting_qa'])

    with col4:
        st.metric("In Progress", stats['in_progress'])

    with col5:
        st.metric("Pending Sign-Offs", stats['pending_sign_offs'])

    with col6:
        st.metric("Open QA Issues", stats['open_qa_issues'], delta=None, delta_color="inverse")

    st.divider()

    # Filters
    st.markdown("### 🔍 Filters")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        status_filter = st.multiselect(
            "Status",
            ["pending", "in_progress", "completed", "qa_approved", "blocked"],
            default=None
        )

    with filter_col2:
        priority_filter = st.multiselect(
            "Priority",
            ["critical", "high", "medium", "low"],
            default=None
        )

    with filter_col3:
        qa_status_filter = st.selectbox(
            "QA Status",
            ["All", "Approved", "Awaiting Review", "Has Issues", "Ready to Finalize"],
            index=0
        )

    with filter_col4:
        show_legion = st.checkbox("Show Legion Tasks Only", value=False)

    # Get all tasks
    all_tasks = manager.get_all_tasks_with_qa_details(limit=200)

    # Apply filters
    filtered_tasks = all_tasks

    if status_filter:
        filtered_tasks = [t for t in filtered_tasks if t['status'] in status_filter]

    if priority_filter:
        filtered_tasks = [t for t in filtered_tasks if t['priority'] in priority_filter]

    if qa_status_filter != "All":
        if qa_status_filter == "Approved":
            filtered_tasks = [t for t in filtered_tasks if t['status'] == 'qa_approved']
        elif qa_status_filter == "Awaiting Review":
            filtered_tasks = [t for t in filtered_tasks if t['status'] == 'completed' and t['sign_offs_pending'] > 0]
        elif qa_status_filter == "Has Issues":
            filtered_tasks = [t for t in filtered_tasks if t['open_qa_issues'] > 0 or t['sign_offs_rejected'] > 0]
        elif qa_status_filter == "Ready to Finalize":
            filtered_tasks = [t for t in filtered_tasks if t['ready_to_finalize'] and t['status'] != 'qa_approved']

    if show_legion:
        filtered_tasks = [t for t in filtered_tasks if t.get('metadata') and t['metadata'].get('legion_task_id')]

    st.divider()

    # Display tasks
    st.markdown(f"### 📋 Tasks ({len(filtered_tasks)} shown)")

    if filtered_tasks:
        for task in filtered_tasks:
            render_task_card(task)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("No tasks match the selected filters")

    # Footer with links
    st.divider()
    st.caption("💡 **Tip:** Tasks automatically trigger QA review when completed. Use TaskCompletionWithQA to ensure proper QA workflow.")
    st.caption(f"📊 **Dashboard:** [View Task Management Dashboard](http://localhost:8505)")


if __name__ == "__main__":
    main()
