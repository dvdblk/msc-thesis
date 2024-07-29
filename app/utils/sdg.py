import matplotlib.colors as colors


def get_sdg_colormap(sdg: int) -> colors.LinearSegmentedColormap:
    # Define SDG colors
    sdg_colors = {
        1: "#E5243B",
        2: "#DDA63A",
        3: "#4C9F38",
        4: "#C5192D",
        5: "#FF3A21",
        6: "#26BDE2",
        7: "#FCC30B",
        8: "#A21942",
        9: "#FD6925",
        10: "#DD1367",
        11: "#FD9D24",
        12: "#BF8B2E",
        13: "#3F7E44",
        14: "#0A97D9",
        15: "#56C02B",
        16: "#00689D",
        17: "#19486A",
    }

    if sdg not in sdg_colors:
        raise ValueError(f"Invalid SDG number. Must be between 1 and 17, got {sdg}")

    sdg_color = sdg_colors[sdg]
    name = f"sdg_{sdg}"

    # Create the colormap
    cmap = colors.LinearSegmentedColormap.from_list(
        name, ["#D3D3D3", "white", sdg_color], N=256
    )
    return cmap
