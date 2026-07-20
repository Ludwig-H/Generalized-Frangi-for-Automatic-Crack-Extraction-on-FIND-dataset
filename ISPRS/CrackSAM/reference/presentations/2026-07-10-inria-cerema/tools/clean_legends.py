from pathlib import Path


MAIN_TEX = Path(__file__).resolve().parents[1] / "source" / "main.tex"

with MAIN_TEX.open('r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    if i + 1 < len(lines) and ('\\textbf{Observations}' in lines[i+1] or '\\textbf{Légende}' in lines[i+1]):
        if '\\vskip' in line:
            i += 2
            continue
    new_lines.append(line)
    i += 1

with MAIN_TEX.open('w') as f:
    f.writelines(new_lines)

print("Successfully removed Observations and Legends!")
