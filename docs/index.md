# Using pandas for NMR data

## Loading data

``` Python
from nmr_to_pandas import nmr_to_pandas
# load a pseudo 3D dataset. `pseudo_dim=0` sets the pseudo dimension assuming
# your data has shape (Z,Y,X)
data = nmr_to_pandas("test_pipe.ft2", pseudo_dim=0)
# your data object now has the attribute `df`
data.df.head()
```

You now have your NMR data in the form of a pandas dataframe where `Z` is the plane number, `Y` and `X` are points

``` console
     Z    Y    X     intensity       Y_PPM      X_PPM
0  0.0  0.0  0.0 -24273.875000  130.538386  10.498205
1  0.0  0.0  1.0 -33351.800781  130.538386  10.490872
2  0.0  0.0  2.0  -9466.987305  130.538386  10.483539
3  0.0  0.0  3.0  -1997.709473  130.538386  10.476206
4  0.0  0.0  4.0  10106.075195  130.538386  10.468873
```

To slice out data you can use the `pandas.Dataframe.query` method.

``` Python
data.df.query("Z==1 & Y_PPM < 120 and X_PPM < 9")
```

``` console
          Z      Y      X     intensity       Y_PPM     X_PPM
201679  1.0  113.0  205.0  32269.908203  119.945665  8.994906
201680  1.0  113.0  206.0  73720.929688  119.945665  8.987572
201681  1.0  113.0  207.0 -35410.578125  119.945665  8.980239
201682  1.0  113.0  208.0    715.999634  119.945665  8.972906
201683  1.0  113.0  209.0  37729.289062  119.945665  8.965573
```

## Plotting

