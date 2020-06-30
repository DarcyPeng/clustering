import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.colors
from scipy.stats import gaussian_kde

from src.utils import DataIO


def factor_scatter_matrix(df, factor, palette=None):
    '''Create a scatter matrix of the variables in df, with differently colored
    points depending on the value of df[factor].
    inputs:
        df: pandas.DataFrame containing the columns to be plotted, as well 
            as factor.
        factor: string or pandas.Series. The column indicating which group 
            each row belongs to.
        palette: A list of hex codes, at least as long as the number of groups.
            If omitted, a predefined palette will be used, but it only includes
            9 groups.
    '''
    if isinstance(factor, str):
        factor_name = factor #save off the name
        factor = df[factor] #extract column
        df = df.drop(factor_name,axis=1) # remove from df, so it 
        # doesn't get a row and col in the plot.

    classes = list(set(factor))

    if palette is None:
        palette = ['#e41a1c', '#377eb8', '#4eae4b', 
                   '#fdfc33', '#ff8101', '#994fa1', 
                   '#a8572c', '#f482be', '#999999']

    color_map = dict(zip(classes,palette))

    if len(classes) > len(palette):
        raise ValueError("Too many groups for the number of colors provided."
                         "We only have {} colors in the palette, but you have {}"
                         "groups.".format(len(palette), len(classes)))

    colors = factor.apply(lambda group: color_map[group])
    axarr = scatter_matrix(df, alpha=1.0, figsize=(10,10),marker='o',c=colors,diagonal=None)

    for rc in range(len(df.columns)):
        for group in classes:
            y = df[factor == group].iloc[:, rc].values
            gkde = gaussian_kde(y)
            ind = np.linspace(y.min(), y.max(), 1000)
            axarr[rc][rc].plot(ind, gkde.evaluate(ind),c=color_map[group])

    return axarr, color_map

def main(ds, folder, separator=";"):
    dataset = Path(f"./{folder}/{ds}.dat")
    dpSeq, labelSeq = DataIO.loadCsvWithIntLabelsAsSeq(dataset, separator=separator)

    df = pd.DataFrame(dpSeq[:, :5])
    cl = pd.Series(labelSeq)
    factor_scatter_matrix(df, cl)
    plt.show()

if __name__ == "__main__":
    #args = ["iris", "datasets", ","]
    args = ["ecoli_result", "result", ";"]
    #args = ["sample_result", "result", ";"]
    main(*args)
