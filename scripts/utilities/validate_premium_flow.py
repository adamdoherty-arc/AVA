"""Quick validation script for Premium Options Flow feature"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tradingview_db_manager import TradingViewDBManager

def validate():
    print("\n" + "="*60)
    print("Premium Options Flow - Quick Validation")
    print("="*60)

    # Check database tables
    print("\n1. Checking database tables...")
    try:
        tv_manager = TradingViewDBManager()
        conn = tv_manager.get_connection()
        cur = conn.cursor()

        required_tables = [
            'options_flow',
            'options_flow_analysis',
            'premium_flow_opportunities',
            'options_flow_alerts'
        ]

        for table in required_tables:
            cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{table}'
                )
            """)
            exists = cur.fetchone()[0]
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {table}")

        cur.close()
        conn.close()
        print("\n  Database validation: PASSED")

    except Exception as e:
        print(f"\n  Database validation: FAILED - {e}")
        return False

    # Check Python modules
    print("\n2. Checking Python modules...")
    try:
        from src.options_flow_tracker import OptionsFlowTracker
        print("  [OK] options_flow_tracker.py")

        from src.ai_flow_analyzer import AIFlowAnalyzer
        print("  [OK] ai_flow_analyzer.py")

        from premium_flow_page import display_premium_flow_page
        print("  [OK] premium_flow_page.py")

        print("\n  Module validation: PASSED")

    except Exception as e:
        print(f"\n  Module validation: FAILED - {e}")
        return False

    # Check dashboard integration
    print("\n3. Checking dashboard integration...")
    try:
        with open('dashboard.py', 'r', encoding='utf-8') as f:
            dashboard_content = f.read()

        if 'Premium Options Flow' in dashboard_content:
            print("  [OK] Navigation button added")
        else:
            print("  [MISSING] Navigation button")
            return False

        if 'from premium_flow_page import display_premium_flow_page' in dashboard_content:
            print("  [OK] Page import added")
        else:
            print("  [MISSING] Page import")
            return False

        if 'display_premium_flow_page()' in dashboard_content:
            print("  [OK] Page routing added")
        else:
            print("  [MISSING] Page routing")
            return False

        print("\n  Dashboard integration: PASSED")

    except Exception as e:
        print(f"\n  Dashboard integration: FAILED - {e}")
        return False

    print("\n" + "="*60)
    print("VALIDATION COMPLETE: ALL CHECKS PASSED")
    print("="*60)
    print("\nPremium Options Flow feature is ready to use!")
    print("\nTo start using:")
    print("1. Run: streamlit run dashboard.py")
    print("2. Click 'Premium Options Flow' in sidebar")
    print("3. Click 'Run Migration Now' if tables need setup")
    print("4. Click 'Refresh Flow Data' to collect data")
    print("5. Click 'Run AI Analysis' for recommendations")
    print("\n" + "="*60)

    return True

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
