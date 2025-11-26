"""
Test script to verify all critical imports after restoration
"""

import sys

def test_import(module_path, class_name):
    """Test importing a module"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        getattr(module, class_name)
        print(f"[PASS] {module_path}.{class_name}")
        return True
    except Exception as e:
        print(f"[FAIL] {module_path}.{class_name}: {e}")
        return False

def main():
    """Run all import tests"""
    print("=" * 60)
    print("Testing Critical Module Imports")
    print("=" * 60)
    print()

    tests = [
        # Phase 1-2: Core Database & Service Layer
        ("src.kalshi_db_manager", "KalshiDBManager"),
        ("src.zone_database_manager", "ZoneDatabaseManager"),
        ("src.services.robinhood_client", "RobinhoodClient"),
        ("src.services.connector_base", "BaseConnector"),
        ("src.services.positions_connector", "PositionsConnector"),
        ("src.services.sync_log_service", "SyncLogService"),
        ("src.services.sync_status_service", "SyncStatusService"),

        # Phase 3: AVA Core
        ("src.ava.nlp_handler", "NaturalLanguageHandler"),
        ("src.ava.telegram_bot", "TelegramBot"),
        ("src.ava.autonomous_agent", "AutonomousAgent"),
        ("src.ava.state_manager", "StateManager"),
        ("src.ava.magnus_integration", "MagnusIntegration"),

        # Phase 4: Sports Betting
        ("src.nfl_db_manager", "NFLDBManager"),
        ("src.kalshi_client_v2", "KalshiClientV2"),
        ("src.nfl_realtime_sync", "NFLRealtimeSync"),
        ("src.enhanced_sports_predictor", "EnhancedSportsPredictor"),

        # Phase 5: Technical Analysis
        ("src.advanced_technical_indicators", "AdvancedTechnicalIndicators"),
        ("src.zone_analyzer", "ZoneAnalyzer"),
        ("src.smart_money_indicators", "SmartMoneyIndicators"),

        # Phase 7: Portfolio Management
        ("src.portfolio_balance_tracker", "PortfolioBalanceTracker"),
        ("src.bankroll_manager", "BankrollManager"),
        ("src.task_manager", "TaskManager"),
    ]

    passed = 0
    failed = 0

    for module_path, class_name in tests:
        if test_import(module_path, class_name):
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
