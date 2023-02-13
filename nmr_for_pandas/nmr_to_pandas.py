from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from skimage.filters import threshold_otsu


@dataclass
class nmrData:
    df: pd.DataFrame
    path: Union[str, Path]
    udic: dict
    dic: dict
    data: np.array
    ndim: int
    uc_dic: dict
    pseudo_dim: Optional[int] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.df = self.df.copy()

    def plot_contour(
        self,
        ax,
        query: Optional[str] = None,
        axis_labels=True,
        invert_axes=False,
        show_cs=False,
        show=False,
        threshold: Union[str, float, None] = "otsu",
        kwargs={},
    ):
        """
        Make contour plot of spectral data.

        Works for 2D data only. If you have a 3D or a pseudo 2D dataset the you must use the query string
        to select the plane that you wish to plot

        For example if you wish to plot the first plane of a 3D data set:
            data.plot_contour(ax, query="Z==0")

        You could also slice the data in X and Y with the following query:
            query = "Z==0 & X_PPM > 7 & X_PPM < 9.5 & Y_PPM > 110 & Y_PPM < 125"
            data.plot_contour(ax, query=query)
        This would plot data where `X_PPM` is less than 9.5 and greater than 7 and `Y_PPM` is less than 125 and greater than 110.

        Parameters
        ----------
        ax : Matplotlib Axes
            The Matplotlib Axes object where the plot will be drawn.
        query : Optional[str], optional
            The query string used to filter the data before plotting. If None, the full data is used.
            See Pandas docs for information on using the query function.
        kwargs : dict, optional
            A dictionary of keyword arguments to be passed to the plotting function.

            nstd : int, optional
                The number of standard deviations used to calculate the threshold for contour plotting.
                Default is 5 However if `threshold` is set to "ostu" then the skimage threshold_otsu function is used
            contour_num : int, optional
                The number of contour levels to be plotted.
                Default is 10.
            contour_factor : float, optional
                The factor used to determine the spacing between contour levels.
                Default is 1.2.
            colors : str
                set the contour line colors (takes precedence of cmap)
            cmap : str
                matplotlib colormap name. If colors is set then this is not used.

        axis_labels : bool
            Automatically generate axis labels based on udic dim labels
        invert_axes : bool
            Invert X and Y axes for NMR conventions
        show_cs : bool
            Annotate the figure with the contour start threshold
            Default is False
        show : bool
            Whether to show an interactive matplotlib figure
            Default is False
        threshold : "otsu" | float | None
            The threshold above which contours at plotted. If "otsu" then the otsu method
            for image thresholding is used (see scikit-image docs). Otherwise the threshold 
            is explicitly set using a float value. If None then the threshold is calculated 
            using the `nstd` value in the kwargs dict. By default this is set to 5 standard 
            deviations over the median intensity value.
            Default is "otsu"

        Returns
        -------
        ax : Matplotlib Axes
            The updated Matplotlib Axes object containing the contour plot.


        Example
        -------

        """
        if query is None:
            df = self.df
        else:
            df = self.df.query(query)

        if self.ndim == 1:
            raise "You only have 1 dimensional data!"
        else:
            pass
        # get limits for ppm scale
        X_min, X_max = df.X_PPM.min(), df.X_PPM.max()
        Y_min, Y_max = df.Y_PPM.min(), df.Y_PPM.max()

        if threshold == "otsu":
            self.threshold = threshold_otsu(df.intensity)
        elif type(threshold) == float:
            self.threshold = threshold
        else:
            self.threshold = calc_threshold(df.intensity, nstd=kwargs.get("nstd", 5))

        contour_kwargs = set_contour_kwargs(kwargs)
        if kwargs.get("negative_contours"):
            plot_negative_contours = True
            neg_contour_kwargs = kwargs.get("negative_contours")
            neg_contour_kwargs = set_contour_kwargs(neg_contour_kwargs)
        else:
            plot_negative_contours = False

        if plot_negative_contours:
            ax.contour(
                -1.0
                * df.intensity.values.reshape(
                    len(df.Y_PPM.unique()), len(df.X_PPM.unique())
                ),
                calc_contour_levels(
                    contour_start=self.threshold,
                    contour_num=kwargs.get("contour_num", 10),
                    contour_factor=kwargs.get("contour_factor", 1.2),
                ),
                extent=(X_max, X_min, Y_max, Y_min),
                **neg_contour_kwargs,
            )
        # create figure
        ax.contour(
            df.intensity.values.reshape(len(df.Y_PPM.unique()), len(df.X_PPM.unique())),
            calc_contour_levels(
                contour_start=self.threshold,
                contour_num=kwargs.get("contour_num", 10),
                contour_factor=kwargs.get("contour_factor", 1.2),
            ),
            extent=(X_max, X_min, Y_max, Y_min),
            **contour_kwargs,
        )
        if axis_labels:
            ax.set_xlabel(f'{self.udic[self.uc_dic["X"]["dim"]]["label"]} ppm')
            ax.set_ylabel(f'{self.udic[self.uc_dic["Y"]["dim"]]["label"]} ppm')

        if invert_axes:
            ax.invert_xaxis()
            ax.invert_yaxis()

        if self.name is not None:
            if contour_kwargs.get("colors"):
                if show_cs:
                    self.name = self.name + f" (cs={self.threshold:.2e})"
                # hack for legends
                ax.plot([],[],color=contour_kwargs.get("colors")[0], label=self.name)
                ax.legend(loc="lower center",bbox_to_anchor=(0.5,1.05))
        return ax


def set_contour_kwargs(kwargs, contour_kwargs={}):
    contour_kwargs = contour_kwargs.copy()
    if kwargs.get("colors"):
        contour_kwargs["colors"] = kwargs.get("colors")
    elif kwargs.get("cmap"):
        contour_kwargs["cmap"] = kwargs.get("cmap", "viridis")
    else:
        contour_kwargs = contour_kwargs
    contour_kwargs["linewidths"] = kwargs.get("linewidths", 0.5)
    return contour_kwargs


def nmr_to_pandas(
    path: Union[str, Path], pseudo_dim: Optional[int] = None, verbose_mode: bool = False
):
    """Convert NMR data to a pandas dataframe.

    This function reads NMR data from a pipe file, guesses the number of dimensions in the data, and converts the data into a pandas dataframe. Points are converted to PPM values using nmrglue unit conversion objects. If a pseudo-dimension is specified, the corresponding column will not be transformed into PPM values.

    Parameters
    ----------
    path : str | Path
        The path to the nmrPipe file containing the data.
    pseudo_dim : int | None, optional
        The index of the pseudo-dimension, if any. For example, if the Z dimension is a pseudo dimension
        pseudo_dim = 0. The default is None.
    verbose_mode : bool, optional
        Whether to display additional information while running the function. The default is False.

    Returns
    -------
    nmrData : nmrData
        A dataclass containing NMR data
    """
    dic, data = ng.pipe.read(path)
    udic = ng.pipe.guess_udic(dic, data)
    ndim = udic["ndim"]
    if verbose_mode:
        print("Data has shape ", data.shape)

    columns = ["Z", "Y", "X", "intensity"]
    if ndim == 1:
        new_A = np.array([(i_x, i) for i_x, i in enumerate(data)])
        columns = columns[2:]

    elif ndim == 2:
        new_A = np.array(
            [(i_y, i_x, x) for i_y, y in enumerate(data) for i_x, x in enumerate(y)]
        )
        columns = columns[1:]

    elif ndim == 3:
        new_A = np.array(
            [
                (i_z, i_y, i_x, x)
                for i_z, z in enumerate(data)
                for i_y, y in enumerate(z)
                for i_x, x in enumerate(y)
            ]
        )
        columns = columns

    df = pd.DataFrame(new_A, columns=columns)
    uc_dic = {}
    for dim, col in enumerate(columns):
        if dim == pseudo_dim:
            print("Skipping pseudo dimension: ", dim)
        else:
            try:
                uc = ng.pipe.make_uc(dic, data, dim=dim)
                df[f"{col}_PPM"] = df[col].apply(uc.ppm)
                uc_dic[col] = dict(uc=uc, dim=dim)
            except:
                pass
    return nmrData(
        df=df,
        path=path,
        pseudo_dim=pseudo_dim,
        udic=udic,
        dic=dic,
        data=data,
        uc_dic=uc_dic,
        ndim=ndim,
    )


def calc_contour_levels(contour_start=30000, contour_num=20, contour_factor=1.2):
    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)
    return cl


def calc_threshold(intensities: pd.Series, nstd: float = 5) -> float:
    threshold = intensities.median() + intensities.std() * nstd
    return threshold


def get_color_iterator(qualitative_cm="Set1"):
    """create an iterator for a qualitative matplotlib colormap
    next(colors)
    """
    colors = plt.cm.get_cmap(qualitative_cm)
    return iter([colors(i) for i in range(20)])


if __name__ == "__main__":
    nmrdata = nmr_to_pandas(path="test_pipe.ft2", pseudo_dim=0)
    print(nmrdata.df.head())
    nmrdata.df.to_pickle("spectrum.pkl")
