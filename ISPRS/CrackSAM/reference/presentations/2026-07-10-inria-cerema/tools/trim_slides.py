from pathlib import Path


MAIN_TEX = Path(__file__).resolve().parents[1] / "source" / "main.tex"

with MAIN_TEX.open('r') as f:
    lines = f.readlines()

idx1 = idx2 = idx3 = idx4 = -1
for i, line in enumerate(lines):
    if 'Performances en \\emph{Zéro-Shot}' in line:
        for j in range(i, len(lines)):
            if '\\end{frame}' in lines[j]:
                idx1 = j
                break
    if '\\section{Résultats sur VT-GraF}' in line:
        idx2 = i
    if 'Fissure 5 : Extraction et Alignement' in line:
        for j in range(i, len(lines)):
            if '\\end{frame}' in lines[j]:
                idx3 = j
                break
    if '\\frame{\\merci}' in line:
        idx4 = i

if idx1 != -1 and idx2 != -1 and idx3 != -1 and idx4 != -1:
    chunk1 = lines[:idx1+1]
    chunk2 = lines[idx2:idx3+1]
    chunk3 = lines[idx4:]
    
    new_lines = chunk1 + ['\n\n'] + chunk2 + ['\n\n'] + chunk3
    with MAIN_TEX.open('w') as f:
        f.writelines(new_lines)
    print("Successfully trimmed main.tex!")
else:
    print(f"Error: indices not found correctly. idx1={idx1}, idx2={idx2}, idx3={idx3}, idx4={idx4}")
