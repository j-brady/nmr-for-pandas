from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

import pandas as pd
import nmrglue as ng
import numpy as np
from rich import print

@dataclass
class nmrData:
    df: pd.DataFrame
    path: Union[str,Path]
    udic: dict
    dic: dict
    data: np.array
    ndim: int
    uc_dic: dict
    pseudo_dim: Optional[int] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.df = self.df.copy()


def nmr_to_pandas(
    path: Union[str,Path], pseudo_dim: Optional[int] = None, verbose_mode: bool = False
):
    """Convert NMR data to a pandas dataframe.

    This function reads NMR data from a pipe file, guesses the number of dimensions in the data, and converts the data into a pandas dataframe. If the number of dimensions is 1, 2, or 3, the corresponding columns in the dataframe are named "Z", "Y", "X", and "intensity", in that order. If a pseudo-dimension is specified, the corresponding column will not be transformed into PPM values.

    Parameters
    ----------
    path : Path
        The path to the pipe file containing the NMR data.
    pseudo_dim : int | None, optional
        The index of the pseudo-dimension, if any. The default is None.
    verbose_mode : bool, optional
        Whether to display additional information while running the function. The default is False.

    Returns
    -------
    df : pandas.DataFrame
        The NMR data as a pandas dataframe.
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
                uc_dic[col] = dict(uc=uc,dim=dim)
            except:
                pass
    return nmrData(df=df,path=path,pseudo_dim=pseudo_dim,udic=udic,dic=dic,data=data,uc_dic=uc_dic,ndim=ndim)

if __name__ == "__main__":
    nmrdata = nmr_to_pandas(path="test_pipe.ft2", pseudo_dim=0)
    print(nmrdata.df.head())
    nmrdata.df.to_pickle("spectrum.pkl")
