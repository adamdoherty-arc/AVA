"""
End-to-End Test of Batch Analysis
Tests the full workflow from database query to AI scoring
"""
import sys
sys.path.insert(0, r'c:\code\Magnus')

from src.ai_options_agent.ai_options_db_manager import AIOptionsDBManager
from src.ai_options_agent.options_analysis_agent import OptionsAnalysisAgent
from src.ai_options_agent.llm_manager import get_llm_manager

print('='*60)
print('END-TO-END BATCH ANALYSIS TEST')
print('='*60)

print('\n[1/4] Initializing components...')
db_manager = AIOptionsDBManager()
llm_manager = get_llm_manager()
agent = OptionsAnalysisAgent(llm_manager=llm_manager)

print('\n[2/4] Querying database for opportunities...')
opportunities = db_manager.get_opportunities(
    dte_range=(5, 45),
    delta_range=(-0.45, -0.15),
    min_premium=100,
    limit=5  # Test with just 5 to speed up
)

print(f'Found {len(opportunities)} opportunities from database')
if len(opportunities) == 0:
    print('ERROR: No opportunities found! Check database has data.')
    sys.exit(1)

print('\n[3/4] Sample opportunities:')
for i, opp in enumerate(opportunities, 1):
    print(f"{i}. {opp['symbol']}: Strike ${opp['strike_price']:.2f}, Premium ${opp['premium']:.2f}, DTE {opp['dte']}")

print('\n[4/4] Running AI analysis on opportunities (WITHOUT LLM for speed)...')
try:
    results = agent.analyze_all_stocks(
        dte_range=(5, 45),
        delta_range=(-0.45, -0.15),
        min_premium=100,
        limit=5,
        use_llm=False  # Skip LLM for speed
    )

    print(f'\nAnalysis complete! Analyzed {len(results)} opportunities')
    print('\nTop 3 Results:')
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. {result.get('symbol', 'N/A')}")
        print(f"   Overall Score: {result.get('overall_score', 0):.1f}/100")
        print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 0):.0f}%")
        print(f"   Strike: ${result.get('strike_price', 0):.2f}")
        print(f"   Premium: ${result.get('premium', 0):.2f}")
        print(f"   Annual Return: {result.get('annual_return', 0):.1f}%")

    print('\n' + '='*60)
    print('TEST PASSED: Batch analysis working correctly!')
    print('='*60)

except Exception as e:
    print(f'\nERROR during analysis: {e}')
    import traceback
    traceback.print_exc()
    print('\n' + '='*60)
    print('TEST FAILED!')
    print('='*60)
    sys.exit(1)
