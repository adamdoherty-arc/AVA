"""Fix the espn_kalshi_matcher.py SQL query bug"""
import re

# Read the file
with open('src/espn_kalshi_matcher.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Handle datetime objects in game_time parsing
old_parse_code = """        # Extract date from game_time (format: "YYYY-MM-DD HH:MM")
        try:
            if game_time:
                game_date = datetime.strptime(game_time[:10], '%Y-%m-%d').date()
            else:
                # If no game time, use today
                game_date = datetime.now().date()
        except:
            game_date = datetime.now().date()"""

new_parse_code = """        # Extract date from game_time (can be datetime object or string)
        try:
            if game_time:
                if isinstance(game_time, datetime):
                    game_date = game_time.date()
                elif isinstance(game_time, str):
                    game_date = datetime.strptime(game_time[:10], '%Y-%m-%d').date()
                else:
                    game_date = datetime.now().date()
            else:
                # If no game time, use today
                game_date = datetime.now().date()
        except Exception as e:
            logger.warning(f"Could not parse game_time {game_time}: {e}")
            game_date = datetime.now().date()"""

content = content.replace(old_parse_code, new_parse_code)

# Fix 2: Remove ::timestamp cast from %s parameters in SQL query
# This is causing "tuple index out of range" error
content = content.replace(
    "%s::timestamp",
    "%s"
)

# Fix 3: Add exc_info to error logging
content = content.replace(
    'logger.error(f"Error matching game to Kalshi: {e}")',
    'logger.error(f"Error matching game to Kalshi: {e}", exc_info=True)'
)

# Write the fixed file
with open('src/espn_kalshi_matcher.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed espn_kalshi_matcher.py")
print("  - Added datetime object handling")
print("  - Fixed SQL parameter casting")
print("  - Enhanced error logging")
