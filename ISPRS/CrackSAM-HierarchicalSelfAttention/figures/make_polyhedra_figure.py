"""Illustrate the proposed reader; no dataset, model or measured output.

K=2 example adapted from thesis §6.1–6.2 (ABC / CD / DEF).
The seven edge identities are leaves; shared points C/D remain visible.
Run this script to regenerate the transparent SVG and PNG next to it.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "svg.fonttype": "none", "svg.hashsalt": "lecteur-polyedres",
    "mathtext.fontset": "dejavusans",
})
NAVY, BLUE, TEAL, GOLD = "#384257", "#355C9B", "#008A8C", "#B88019"
GRAY, LIGHT = "#707A89", "#CCD1D8"
fig, ax = plt.subplots(figsize=(16, 9))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 16), ylim=(0, 9), aspect="equal")
ax.axis("off")


def label(x, y, text, size=14, color=NAVY, weight="normal", **kw):
    return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                   ha=kw.pop("ha", "center"), va="center", **kw)


def line(points, color=NAVY, width=1.8, **kw):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=width, solid_capstyle="round", **kw)


def arrow(start, end, color=NAVY, width=1.7, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                 mutation_scale=16, lw=width, color=color, **kw))


def box(x, y, w, h, text, color=NAVY, size=15):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                 boxstyle="round,pad=0.025,rounding_size=0.09",
                 edgecolor=color, facecolor="none", lw=1.6))
    label(x, y, text, size=size, color=color)


label(8, 8.52, "Lire les objets et leurs regroupements", 25, weight="bold")
label(8, 8.06, "Exemple K=2 : arêtes reliées par triangles", 16, color=GRAY)
for x, title in [(2.45, "1. Les objets géométriques"),
                 (7.65, "2. Leur hiérarchie"),
                 (12.98, "3. Un petit lecteur")]:
    label(x, 7.40, title, 17, weight="bold")
line([(4.80, 2.02), (4.80, 7.02)], LIGHT, 1)
line([(10.12, 2.02), (10.12, 7.02)], LIGHT, 1)

# Seven atomic edges. C and D are shared by two point supports.
pts = {"A": (0.83, 6.10), "B": (0.83, 4.65), "C": (2.10, 5.375),
       "D": (2.90, 5.375), "E": (4.17, 6.10), "F": (4.17, 4.65)}
for pair, col in [("AB", BLUE), ("BC", BLUE), ("AC", BLUE),
                  ("CD", GOLD), ("DE", TEAL), ("EF", TEAL), ("DF", TEAL)]:
    line([pts[pair[0]], pts[pair[1]]], col, 2.7)
for p, (x, y) in pts.items():
    shared = p in "CD"
    if shared:
        ax.add_patch(Circle((x, y), 0.12, edgecolor=GOLD,
                           facecolor="none", lw=1.5))
    ax.add_patch(Circle((x, y), 0.045, color=NAVY, zorder=4))
    dy = 0.28 if p in "AE" else -0.28
    label(x, y+dy, p, 15, weight="bold")
for x, y, text, col in [
    (0.54, 5.375, "AB", BLUE), (1.38, 4.76, "BC", BLUE),
    (1.38, 5.99, "AC", BLUE), (2.50, 5.68, "CD", GOLD),
    (3.63, 5.99, "DE", TEAL), (4.48, 5.375, "EF", TEAL),
    (3.63, 4.76, "DF", TEAL),
]:
    label(x, y, text, 12, color=col)
label(1.25, 4.00, "AB · BC · AC", 14, color=BLUE, weight="bold")
label(2.50, 4.00, "CD", 14, color=GOLD, weight="bold")
label(3.75, 4.00, "DE · EF · DF", 14, color=TEAL, weight="bold")
label(2.50, 3.56, "Trois groupes au niveau r₁", 14)
label(2.50, 2.70, "C et D sont partagés.\nLes arêtes restent identifiées.", 15)

# Partial seven-leaf forest: no later fusion without additional connectors.
xs = [5.72, 6.26, 6.80, 7.60, 8.40, 8.94, 9.48]
yleaf, r1, ytop = 4.22, 5.46, 6.58
names = ["AB", "BC", "AC", "CD", "DE", "EF", "DF"]
colors = [BLUE]*3+[GOLD]+[TEAL]*3
for i, (x, name, col) in enumerate(zip(xs, names, colors)):
    line([(x, yleaf), (x, ytop if i == 3 else r1)], col, 1.9)
    ax.add_patch(Circle((x, yleaf), .043, color=col))
    label(x, yleaf-.32, name, 12, color=col, weight="bold")
for start, end, center, col in [(xs[0], xs[2], xs[1], BLUE),
                               (xs[4], xs[6], xs[5], TEAL)]:
    line([(start, r1), (end, r1)], col, 2.3)
    line([(center, r1), (center, ytop)], col, 1.9)
    ax.add_patch(Circle((center, r1), .058, color=col))
label(5.23, r1, "r₁", 15, color=GRAY)
label(7.60, 6.88, "Extrait : trois groupes distincts", 13, color=GRAY)
label(7.60, 3.45, "7 feuilles : les arêtes", 14)
label(7.60, 2.70, "Un objet à chaque nœud ;\nla suite dépend des connecteurs.", 15)

# Frozen feature input and trainable geometric reader. No fictitious prediction.
box(12.90, 6.54, 3.48, .78, "Représentations gelées  fₓ")
arrow((12.90, 6.11), (12.90, 5.69))
box(12.90, 5.11, 3.48, 1.04, "Géométrie + fusions\npetit lecteur appris", TEAL, 15)
arrow((9.53, 5.11), (11.10, 5.11), TEAL)
label(10.42, 5.76, "objets\n+ niveaux", 11, color=TEAL)
arrow((12.90, 4.56), (12.90, 4.13))
box(12.90, 3.62, 3.48, .98, "Vote vers les points incidents\n$w_{x\\tau}=S_\\tau/T_x$", GOLD, 14)
arrow((12.90, 3.10), (12.90, 2.61))
box(12.90, 2.23, 3.48, .68, "Décision par point", NAVY, 15)
# Fine frozen features bypass the geometric summaries.
line([(14.67, 6.54), (15.28, 6.54), (15.28, 2.23)], BLUE, 1.7)
arrow((15.28, 2.23), (14.70, 2.23), BLUE)
label(15.64, 4.46, "Voie fine fₓ conservée", 12, color=BLUE, rotation=90)
label(8, 1.26, "Les points peuvent contribuer à plusieurs objets ;\nla hiérarchie organise les échanges entre ces objets.", 17)
label(8, .43, "Perspective — exemple combinatoire K=2", 12, color=GRAY)

out = Path(__file__).resolve().parent
for suffix in ("svg", "png"):
    metadata = {"Date": None} if suffix == "svg" else None
    target = out / f"lecteur_polyedres.{suffix}"
    fig.savefig(target, dpi=220, transparent=True, metadata=metadata)
    if suffix == "svg":
        target.write_text("\n".join(row.rstrip() for row in
                                   target.read_text().splitlines()) + "\n")
plt.close(fig)
