"""
Registers custom colormaps
"""

from matplotlib import colormaps
from matplotlib.colors import ListedColormap

HYDROCLASS_10 = [
    "#C7C0BF",  # NC  - Not classified (neutral gray)
    "#A6CEE3",  # AG  - Aggregates (light icy blue)
    "#1F78B4",  # CR  - Crystals (deep ice blue)
    "#B2DF8A",  # LR  - Light rain (soft green)
    "#FB9A99",  # RP  - Rimed particles (green → mixed phase)
    "#33A02C",  # RN  - Rain (warm light red)
    "#E31A1C",  # VI  - Vertically oriented ice (strong red)
    "#FDBF6F",  # WS  - Wet snow (yellow–orange, melting)
    "#FF7F00",  # MH  - Melting hail (orange)
    "#6A3D9A",  # IH  - Ice hail (deep purple)
]


hydroclass10 = ListedColormap(HYDROCLASS_10, name="hydroclass10")

colormaps.register(hydroclass10)
