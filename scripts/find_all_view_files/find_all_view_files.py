import os
import json
import sys

log_path = r"C:\Users\opera\.gemini\antigravity-ide\brain\9fc5a517-fc5e-466f-8e5c-83627c60f555\.system_generated\logs\transcript.jsonl"

if not os.path.exists(log_path):
    print("Log not found")
    sys.exit(1)

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for idx, line in enumerate(f, 1):
        if "view_file" in line and "gerar_linha_centro.py" in line:
            try:
                data = json.loads(line)
                # Let's inspect the payload
                print(f"\n--- LOG LINE {idx} ---")
                
                # Helper to print view_file details
                def check_view(obj):
                    if isinstance(obj, dict):
                        if obj.get('AbsolutePath') and 'gerar_linha_centro.py' in obj.get('AbsolutePath'):
                            print(f"  Request view_file: {obj}")
                        for v in obj.values():
                            check_view(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            check_view(item)
                check_view(data)
                
                # Check if there is output in this line
                # Let's see if we can find strings containing code
                code_snippets = []
                def find_code(obj):
                    if isinstance(obj, str):
                        if "def extract_centerline_from_polygon" in obj:
                            code_snippets.append(obj)
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            find_code(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            find_code(item)
                find_code(data)
                for snippet in code_snippets:
                    lines = snippet.split('\n')
                    print(f"  Found code output of {len(lines)} lines. Snippet: {snippet[:200].strip()}...")
                    # Let's write it to a temp file
                    with open(f"view_log_{idx}_{len(lines)}.py", "w", encoding="utf-8") as out:
                        out.write(snippet)
            except Exception as e:
                pass
