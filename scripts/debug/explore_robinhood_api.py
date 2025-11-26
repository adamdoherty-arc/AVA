"""
Explore what data is available from Robinhood API
"""
import robin_stocks.robinhood as rh
import inspect

print("=" * 80)
print("ROBINHOOD API EXPLORATION")
print("=" * 80)

# Get all available modules and functions
print("\nAvailable modules in robin_stocks.robinhood:")
modules = [name for name, obj in inspect.getmembers(rh) if inspect.ismodule(obj)]
for module in sorted(modules):
    print(f"  - {module}")

print("\n" + "=" * 80)
print("AVAILABLE FUNCTIONS:")
print("=" * 80)

# Get all functions
funcs = {}
for name, obj in inspect.getmembers(rh):
    if inspect.isfunction(obj):
        funcs[name] = obj

# Categorize by keywords
categories = {
    'stocks': [],
    'options': [],
    'crypto': [],
    'markets': [],
    'account': [],
    'orders': [],
    'news': [],
    'events': [],
    'sports': [],
    'other': []
}

for func_name in sorted(funcs.keys()):
    categorized = False
    for category in categories.keys():
        if category in func_name.lower():
            categories[category].append(func_name)
            categorized = True
            break
    if not categorized:
        categories['other'].append(func_name)

for category, func_list in categories.items():
    if func_list:
        print(f"\n{category.upper()} ({len(func_list)} functions):")
        for func in func_list:
            sig = inspect.signature(funcs[func])
            print(f"  - {func}{sig}")

# Check for any "event" or "sports" related endpoints
print("\n" + "=" * 80)
print("SEARCHING FOR EVENTS/SPORTS DATA:")
print("=" * 80)

event_related = [name for name in funcs.keys() if 'event' in name.lower() or 'sport' in name.lower() or 'calendar' in name.lower() or 'prediction' in name.lower()]

if event_related:
    print(f"\nFound {len(event_related)} event/sports-related functions:")
    for func in event_related:
        print(f"  - {func}")
else:
    print("\nNo sports or event-related functions found in standard API")

# Check if there are any undocumented endpoints
print("\n" + "=" * 80)
print("CHECKING FOR UNDOCUMENTED ENDPOINTS:")
print("=" * 80)

# Login first to access authenticated endpoints
try:
    print("\nAttempting Robinhood login...")
    import os
    from dotenv import load_dotenv
    load_dotenv()

    username = os.getenv('ROBINHOOD_USERNAME')
    password = os.getenv('ROBINHOOD_PASSWORD')

    if username and password:
        login = rh.login(username, password)
        print("✓ Logged in successfully")

        # Try to access some endpoints that might have events
        print("\nTrying potential event endpoints...")

        # Check market data
        try:
            # Get market movers
            movers = rh.get_top_movers()
            print(f"✓ Market movers available: {len(movers) if movers else 0} stocks")
        except:
            pass

        # Check news
        try:
            news = rh.get_news('AAPL')
            print(f"✓ News available: {len(news) if news else 0} articles")
            if news:
                print(f"  Sample: {news[0].get('title', 'N/A')}")
        except Exception as e:
            print(f"✗ News error: {e}")

        print("\nRobinhood does NOT appear to have sports/events betting data.")
        print("Robinhood focuses on:")
        print("  - Stocks & ETFs")
        print("  - Options")
        print("  - Cryptocurrency")
        print("  - Market news & data")

        rh.logout()
    else:
        print("No Robinhood credentials found in .env")

except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("""
For sports events/betting data, you would need to use a different API such as:
  - The Odds API (https://the-odds-api.com/)
  - Sportradar API
  - ESPN API
  - SportsData.io

Robinhood is a stock/options/crypto trading platform and does not provide
sports betting or events data.
""")
