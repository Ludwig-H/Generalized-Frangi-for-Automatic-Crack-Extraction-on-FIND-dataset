from pathlib import Path


MAIN_TEX = Path(__file__).resolve().parents[1] / "source" / "main.tex"

with MAIN_TEX.open('r') as f:
    lines = f.readlines()

idx_section = -1
idx_merci = -1

for i, line in enumerate(lines):
    if '\\section{Que faire ?..}' in line:
        idx_section = i
    if '\\frame{\\merci}' in line:
        idx_merci = i

if idx_section != -1 and idx_merci != -1:
    # Keep the section header and transition page
    chunk1 = lines[:idx_section+2]
    # Keep the thank you slide and end of document
    chunk2 = lines[idx_merci:]
    
    new_lines = chunk1 + ['\n\n'] + chunk2
    with MAIN_TEX.open('w') as f:
        f.writelines(new_lines)
    print("Successfully trimmed slides under 'Que faire ?..'!")
else:
    print(f"Error: indices not found. idx_section={idx_section}, idx_merci={idx_merci}")
