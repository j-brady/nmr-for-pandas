import shutil
from pathlib import Path
from textwrap import dedent

import typer
from rich import print
import toml
import matplotlib.pyplot as plt

from .nmr_to_pandas import nmr_to_pandas

app = typer.Typer()

def make_toml_file():
    toml = """\
    [output]
    dir = "./"
    fmt = [".pdf",".png"]
    fname = "test"

    [contour]
    linewidths = 0.5
    cmap = "Set1"

    [[spectra]]
    path = "test_pipe.ft2"
    name = "Example 1"
    query = "Z==0"
    pseudo_dim = 0
    show_cs = true

    [[spectra]]
    path = "test_pipe.ft2"
    name = "Example 2"
    query = "Z==0 & X_PPM > 8 & X_PPM < 9"
    pseudo_dim = 0
    show_cs = true
    threshold = 1e7
    """
    return dedent(toml)

@app.command()
def new(name:Path="spectra.toml"):
    if name.exists():
        backup = name.with_suffix(".toml.backup")
        print(f"[yellow]{name} exists... copying to {backup}[/yellow]")
        shutil.copyfile(name,backup)
    else:
        pass
    print(f"[green]Creating {name}[/green]")
    name.write_text(make_toml_file())

@app.command()
def plot(toml_file:Path, show: bool=False):
    config = toml.load(toml_file)
    output = config.get("output",{})
    out_dir = Path(output.get("dir","./"))
    contour_config = config.get("contour",dict(cmap="Set1",linewidths=0.5))
    colors = plt.cm.get_cmap(contour_config.get("cmap"))
    colors = iter([colors(i) for i in range(20)])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for spectrum in config.get("spectra",[]):
        data = nmr_to_pandas(spectrum.get("path"),pseudo_dim=spectrum.get("pseudo_dim",None))
        data.name = spectrum.get("name",None)
        query = spectrum.get("query")
        kwargs=dict(colors=[next(colors)])
        ax = data.plot_contour(ax, query=query, threshold=spectrum.get("threshold","otsu"), kwargs=kwargs, show_cs=spectrum.get("show_cs"))
        
    ax.invert_yaxis()
    ax.invert_xaxis()
    [plt.savefig(out_dir / Path(output.get("fname","test")).with_suffix(fmt), bbox_inches="tight") for fmt in output.get("fmt",[".pdf",".png"])]
    if show:
        plt.show()



if __name__ == "__main__":
    app()
