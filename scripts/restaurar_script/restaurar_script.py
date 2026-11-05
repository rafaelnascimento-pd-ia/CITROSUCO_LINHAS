import os
import json
import sys

log_path = r"C:\Users\opera\.gemini\antigravity-ide\brain\9fc5a517-fc5e-466f-8e5c-83627c60f555\.system_generated\logs\transcript.jsonl"
dest_path = r"c:\BRUNO\PROJETOS\python\projeto_i.a_linhas_citrosuco\scripts\gerar_linha_centro.py"

if not os.path.exists(log_path):
    print(f"Error: Log file not found at {log_path}")
    sys.exit(1)

print("Searching conversation logs for the original script...")

candidates = []
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line_idx, line in enumerate(f, 1):
        if "extract_centerline_from_polygon" in line:
            try:
                data = json.loads(line)
                # Search for all strings inside this JSON object
                def search_obj(obj):
                    if isinstance(obj, str):
                        if "def extract_centerline_from_polygon" in obj:
                            candidates.append((line_idx, len(obj), obj))
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            search_obj(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            search_obj(item)
                search_obj(data)
            except Exception:
                pass

print(f"Found {len(candidates)} potential candidates in the logs.")
candidates.sort(key=lambda x: x[1], reverse=True)

success = False
for idx, (line_num, length, code) in enumerate(candidates):
    print(f"Candidate {idx}: Line {line_num} | Length: {length} chars")
    
    # Check if this code block has truncation markers
    if "truncated" in code or "<truncated" in code:
        print("  -> Truncated, skipping.")
        continue
        
    # Let's clean the lines (remove line numbers like "1: #!/usr/bin/env python")
    lines = code.split("\n")
    cleaned_lines = []
    for l in lines:
        parts = l.split(":", 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            cleaned_lines.append(parts[1])
        else:
            cleaned_lines.append(l)
            
    final_code = "\n".join(cleaned_lines)
    
    # Let's save this as the restored script
    with open(dest_path, "w", encoding="utf-8") as out:
        out.write(final_code)
    print(f"SUCCESS: Restored original script of {len(final_code.split('\n'))} lines to {dest_path}!")
    success = True
    break

if not success:
    # If all candidates in the current log are truncated, let's look for previous backups on disk
    print("No untruncated candidate found in the log. Looking for local backups on disk...")
    # Search all python files in the workspace containing `extract_centerline_from_polygon`
    # and not having DBSCAN or complex merge functions.
    workspace_dir = r"c:\BRUNO\PROJETOS\python\projeto_i.a_linhas_citrosuco"
    possible_backups = []
    for root, dirs, files in os.walk(workspace_dir):
        # Ignore virtual environments
        if ".venv" in root or "venv" in root or ".git" in root:
            continue
        for f in files:
            if f.endswith(".py") and f != "gerar_linha_centro.py" and f != "restaurar_script.py":
                f_path = os.path.join(root, f)
                try:
                    with open(f_path, "r", encoding="utf-8") as file:
                        text = file.read()
                    if "extract_centerline_from_polygon" in text and "DBSCAN" not in text and "merge_aligned_lines" not in text:
                        line_count = len(text.split("\n"))
                        if 180 <= line_count <= 230:
                            possible_backups.append((f_path, line_count, text))
                except Exception:
                    pass
                    
    if possible_backups:
        # Sort by line count close to 211
        possible_backups.sort(key=lambda x: abs(x[1] - 211))
        best_backup_path, best_line_count, best_text = possible_backups[0]
        print(f"Found a backup: {best_backup_path} ({best_line_count} lines). Restoring from it...")
        with open(dest_path, "w", encoding="utf-8") as out:
            out.write(best_text)
        print(f"SUCCESS: Restored from backup {best_backup_path}!")
        success = True
    else:
        print("No local backups found.")

if not success:
    print("Could not find any clean version of the 211-line script.")
