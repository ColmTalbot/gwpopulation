import numpy as np


# coding: utf-8
def circle(radius, phi, size=400, color="black"):
    center = size / 2
    positions = (
        center + radius * np.cos(phi),
        center + radius * np.sin(phi),
        center - radius * np.cos(phi),
        center - radius * np.sin(phi),
        center + radius * np.cos(phi - np.pi / 6),
        center + radius * np.sin(phi - np.pi / 6),
        center - radius * np.cos(phi - np.pi / 6),
        center - radius * np.sin(phi - np.pi / 6),
    )
    width = 2 * radius / 100
    inner = radius / 10
    ends = (
        center + radius * np.cos(phi + np.arctan(1 / 10)),
        center + radius * np.sin(phi + np.arctan(1 / 10)),
        center - radius * np.cos(phi + np.arctan(1 / 10)),
        center - radius * np.sin(phi + np.arctan(1 / 10)),
    )
    return f"""
  <!-- <circle cx="{center}" cy="{center}" r="{radius}" stroke="{color}" stroke-width="2" fill="none" /> -->
  <circle cx="{positions[0]}" cy="{positions[1]}" r="{inner}" stroke="{color}" stroke-width="{width}" fill="none" />
  <circle cx="{positions[2]}" cy="{positions[3]}" r="{inner}" stroke="{color}" stroke-width="{width}" fill="none" />

  <path d="M {ends[0]} {ends[1]} A {radius} {radius} 0 0 1 {positions[6]} {positions[7]}" stroke="{color}" stroke-width="{width}" fill="none" />
  <path d="M {ends[2]} {ends[3]} A {radius} {radius} 0 0 1 {positions[4]} {positions[5]}" stroke="{color}" stroke-width="{width}" fill="none" />
    """


def logo(size=400, color="rgb(40,10,40)", center="rgb(240,240,240)", text=False):

    gradient = f"""  <!-- Define the radial gradient -->
  <defs>
    <radialGradient id="grad1" cx="50%" cy="50%" r="50%" fx="50%" fy="50%">
      <stop offset="00%" style="stop-color:{center};stop-opacity:1" />
      <stop offset="50%" style="stop-color:{center};stop-opacity:1" />
      <stop offset="90%" style="stop-color:{color};stop-opacity:1" />
    </radialGradient>
  </defs>"""
    data = f"""<svg width="{size * 5**int(text)}" height="{size}" xmlns="http://www.w3.org/2000/svg">"""
    #     data += """
    #   {gradient}

    #   <!-- Use the gradient for the background -->
    #   <!-- <rect x="0" y="0" width="100%" height="100%" fill="url(#grad1)" /> -->
    # """
    # data += circle(120, 0.3 * np.pi, size=size)
    largest = size / 4
    data += circle(largest, -0.1 * np.pi, size=size, color=color)
    data += circle(largest / 5, 0.65 * np.pi, size=size, color=color)
    data += circle(largest * 2 / 5, 0.15 * np.pi, size=size, color=color)
    data += circle(largest * 3 / 5, 0.4 * np.pi, size=size, color=color)
    data += circle(largest * 4 / 5, -0.3 * np.pi, size=size, color=color)

    if text:
        data += f"""\t<text x="{size * 0.9}" y="{size * 0.68}" font-family="Verdana" font-size="{size * 0.5}" fill="{color}">GWPopulation</text>"""
    data += "\n</svg>"
    return data


if __name__ == "__main__":
    print(
        logo(size=40, color="rgb(240,240,240)", center="rgb(215,245,215)", text=False),
        file=open("_static/logo-dark.svg", "w"),
    )
    print(
        logo(size=40, color="rgb(15,15,15)", center="rgb(215,245,215)", text=False),
        file=open("_static/logo.svg", "w"),
    )
    print(
        logo(size=400, color="rgb(240,240,240)", center="rgb(215,245,215)", text=True),
        file=open("_static/logo-long-dark.svg", "w"),
    )
    print(
        logo(size=400, color="rgb(15,15,15)", center="rgb(215,245,215)", text=True),
        file=open("_static/logo-long.svg", "w"),
    )
