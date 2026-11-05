import os

print("Scanning recovered files for extract_centerline_from_polygon:")

files = [f for f in os.listdir('.') if (f.startswith('recovered_') or f.startswith('db_match_')) and (f.endswith('.py') or f.endswith('.txt'))]

for f_name in files:
    try:
        with open(f_name, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        # Check if def extract_centerline_from_polygon is in it
        matches = []
        for idx, line in enumerate(lines):
            if "def extract_centerline_from_polygon" in line:
                matches.append((idx + 1, line.strip()))
                
        if matches:
            print(f"File: {f_name} | Total Lines: {len(lines)} | Matches: {matches}")
    except Exception as e:
        print(f"Error reading {f_name}: {e}")
