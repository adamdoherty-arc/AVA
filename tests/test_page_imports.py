"""Test script to check all page imports"""
import sys
import traceback

pages_to_test = [
    ("positions_page_improved", "show_positions_page"),
    ("game_cards_visual_page", "show_game_cards"),
    ("prediction_markets_page", "show_prediction_markets"),
    ("premium_flow_page", "display_premium_flow_page"),
    ("sector_analysis_page", "display_sector_analysis_page"),
    ("earnings_calendar_page", "show_earnings_calendar"),
    ("cache_metrics_page", "show_cache_metrics"),
    ("xtrades_watchlists_page", None),
    ("supply_demand_zones_page", None),
    ("calendar_spreads_page", None),
    ("enhancement_agent_page", "render_enhancement_agent_page"),
]

print("="*60)
print("TESTING PAGE IMPORTS")
print("="*60)

results = {"passed": [], "failed": []}

for module_name, function_name in pages_to_test:
    try:
        module = __import__(module_name)
        if function_name:
            func = getattr(module, function_name, None)
            if func is None:
                results["failed"].append((module_name, f"Function '{function_name}' not found"))
                print(f"FAIL {module_name}.{function_name} - Function not found")
            else:
                results["passed"].append(module_name)
                print(f"PASS {module_name}.{function_name}")
        else:
            results["passed"].append(module_name)
            print(f"PASS {module_name}")
    except ImportError as e:
        results["failed"].append((module_name, str(e)))
        print(f"FAIL {module_name} - ImportError: {e}")
    except Exception as e:
        results["failed"].append((module_name, str(e)))
        print(f"FAIL {module_name} - {type(e).__name__}: {e}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"PASS: {len(results['passed'])}")
print(f"FAIL: {len(results['failed'])}")

if results["failed"]:
    print("\n" + "="*60)
    print("FAILED IMPORTS")
    print("="*60)
    for module, error in results["failed"]:
        print(f"\n{module}:")
        print(f"  {error}")
