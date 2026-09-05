"""Regenerate an illustrative 16:9 diagram with matplotlib; no data/model needed."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
NAVY, ACCENT, GRAY = "#243A59", "#177E89", "#627186"
PALE, TINT, LINE = "#F3F6FA", "#EAF5F5", "#CDD5E0"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "svg.hashsalt": "frangi-hierarchy-perspective"})
fig, ax = plt.subplots(figsize=(16, 9), dpi=180)
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 16), ylim=(0, 9), aspect="equal")
ax.axis("off")


def text(x, y, s, size=15, color=NAVY, weight="normal", ha="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight,
            ha=ha, va="center", linespacing=1.35, **kw)


def box(x, y, w, h, s="", fill="none", edge=LINE, size=15, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, linewidth=1.2,
        boxstyle="round,pad=0.02,rounding_size=0.07", edgecolor=edge, facecolor=fill))
    text(x + w / 2, y + h / 2, s, size=size, weight="bold" if bold else "normal")


def arrow(a, b, color=NAVY, lw=1.8):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                               color=color, linewidth=lw))


# Navy and teal keep the perspective distinct without red blocks.
ax.plot([.55, .55, .88], [8.06, 8.58, 8.58], color=ACCENT, lw=4)
text(1.03, 8.42, "SAM fournit les représentations ;", 26, weight="bold", ha="left")
text(1.03, 7.88, "le graphe Frangi organise leurs relations", 26, weight="bold", ha="left")

box(.55, 2.03, 4.95, 5.16, fill=PALE)
box(5.84, 2.03, 9.60, 5.16)
text(.84, 6.85, "La hiérarchie Frangi", 19, weight="bold", ha="left")
text(6.15, 6.85, "Un biais dans un seul bloc d’attention", 18,
     weight="bold", ha="left")

# Exact merge tree: edge costs are illustrative, not measurements.
x = [.98, 1.79, 2.60, 3.41, 4.22, 5.03]
base, scale = 3.17, 4.0
nodes = {i: (px, base) for i, px in enumerate(x)}
merges = [(0, 1, .10), (3, 4, .15), (6, 2, .20),
          (7, 5, .30), (8, 9, .60)]
for node_id, (left, right, cost) in enumerate(merges, start=6):
    xl, yl = nodes[left]
    xr, yr = nodes[right]
    height = base + scale * cost
    c = ACCENT if cost == .60 else NAVY
    ax.plot([xl, xl, xr, xr], [yl, height, height, yr], color=c, lw=2)
    nodes[node_id] = ((xl + xr) / 2, height)
    text((xl + xr) / 2, height + .17, f"{cost:g}".replace(".", ","), 12, c,
         bbox={"facecolor": PALE, "edgecolor": "none", "pad": .7})
for px, name in zip(x, "abcdef"):
    ax.scatter(px, base, color=NAVY, s=40, zorder=3)
    text(px, 2.93, name, 14)
text(3.03, 6.17, "Coût de fusion illustratif", 12, GRAY)
text(3.03, 2.56, "Feuilles : positions dans l’image", 12)
text(3.03, 2.25, "Branches : groupes emboîtés", 12)

# Image representation flows downward into a modified attention block.
box(6.17, 5.67, 1.38, .72, "Image", size=15)
arrow((7.58, 6.03), (8.08, 6.03))
box(8.12, 5.67, 6.62, .72,
    "SAM 2  ·  caractéristiques avant le bloc choisi", size=14, fill=PALE, bold=True)
arrow((10.37, 5.64), (10.37, 5.02))
box(8.76, 4.03, 3.24, .96, "Attention guidée\ndans l’encodeur", fill=TINT, edge=ACCENT,
    size=16, bold=True)
arrow((12.03, 4.51), (12.50, 4.51))
box(12.54, 4.03, 2.20, .96, "Suite de SAM 2\n+ masque", size=14)

# Explicit relation, not a point prompt: first shared group of each pair.
arrow((5.53, 4.51), (8.72, 4.51), color=ACCENT, lw=2)
text(7.14, 5.06, "Niveaux de fusion\n→ relations entre tokens", 12, ACCENT)
text(10.66, 3.58, "score(i, j) = score SAM(i, j) + α × relation(i, j)",
     14, color=NAVY)
box(6.36, 2.38, 8.55, .78,
    "Tous les poids gelés · aucun token fusionné\nα fixé sur validation ; α = 0 retrouve le modèle initial",
    fill=PALE, edge=LINE, size=13)

text(.61, 1.51, "La hiérarchie organise les échanges ; chaque token est conservé.",
     17, weight="bold", ha="left")
text(.61, 1.08, "Test décisif : les niveaux aident-ils davantage qu’une seule partition ?",
     15, ha="left")
ax.plot([.6, 15.4], [.66, .66], color=LINE, lw=1)
text(.61, .35, "Perspective · SAM 2 gelé", 13, ACCENT, weight="bold", ha="left")
text(15.4, .35, "Arbre illustratif · aucun masque simulé", 11, GRAY, ha="right")
for suffix in ("svg", "png"):
    destination = OUT / f"guidage_hierarchique.{suffix}"
    metadata = {"Date": None} if suffix == "svg" else None
    fig.savefig(destination, transparent=True, dpi=180, metadata=metadata)
    if suffix == "svg":
        destination.write_text("\n".join(line.rstrip() for line in
                               destination.read_text().splitlines()) + "\n")
plt.close(fig)
