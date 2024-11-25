"""Visualize cumulative SHAP values."""

from typing import Union
import matplotlib.cm as cm
import matplotlib.pyplot as pl
import numpy as np
import shap
import warnings
import pandas as pd
from src._legacy import LogitLink, convert_to_link
from src._general import convert_name,encode_array_if_needed,approximate_interactions
import numpy as np
from shap import Explanation
from shap.utils import OpChain
from src._colorconv import lab2rgb, lch2lab
from matplotlib.colors import LinearSegmentedColormap,LogNorm

def hclust_ordering(X, metric="sqeuclidean", anchor_first=False):
    """A leaf ordering is under-defined, this picks the ordering that keeps nearby samples similar."""
    # compute a hierarchical clustering and return the optimal leaf ordering
    D = scipy.spatial.distance.pdist(X, metric)
    cluster_matrix = scipy.cluster.hierarchy.complete(D)
    return scipy.cluster.hierarchy.leaves_list(scipy.cluster.hierarchy.optimal_leaf_ordering(cluster_matrix, D))

def lch2rgb(x):
    return lab2rgb(lch2lab([[x]]))[0][0]

blue_lch = [54., 70., 4.6588]
l_mid = 40.
red_lch = [54., 90., 0.35470565 + 2* np.pi]
gray_lch = [55., 0., 0.]
blue_rgb = lch2rgb(blue_lch)
red_rgb = lch2rgb(red_lch)
gray_rgb = lch2rgb(gray_lch)
white_rgb = np.array([1.,1.,1.])

blue_rgb = lch2rgb(blue_lch)
red_rgb = lch2rgb(red_lch)

colors = []
for alpha in np.linspace(1, 0, 100):
    c = blue_rgb * alpha + (1 - alpha) * white_rgb
    colors.append(c)
for alpha in np.linspace(0, 1, 100):
    c = red_rgb * alpha + (1 - alpha) * white_rgb
    colors.append(c)
red_white_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

reds = []
greens = []
blues = []
alphas = []
nsteps = 100
l_vals = list(np.linspace(blue_lch[0], l_mid, nsteps//2)) + list(np.linspace(l_mid, red_lch[0], nsteps//2))
c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)
for pos,l,c,h in zip(np.linspace(0, 1, nsteps), l_vals, c_vals, h_vals): # noqa: E741
    lch = [l, c, h]
    rgb = lch2rgb(lch)
    reds.append((pos, rgb[0], rgb[0]))
    greens.append((pos, rgb[1], rgb[1]))
    blues.append((pos, rgb[2], rgb[2]))
    alphas.append((pos, 1.0, 1.0))

red_blue = LinearSegmentedColormap('red_blue', {
    "red": reds,
    "green": greens,
    "blue": blues,
    "alpha": alphas
})
red_blue.set_bad(gray_rgb, 1.0)
red_blue.set_over(gray_rgb, 1.0)
red_blue.set_under(gray_rgb, 1.0) # "under" is incorrectly used instead of "bad" in the scatter plot

labels = {
        'MAIN_EFFECT': "SHAP main effect value for\n%s",
        'INTERACTION_VALUE': "SHAP interaction value",
        'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
        'VALUE': "SHAP value (impact on model output)",
        'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
        'VALUE_FOR': "SHAP value for\n%s",
        'PLOT_FOR': "SHAP plot for %s",
        'FEATURE': "Feature %s",
        'FEATURE_VALUE': "Feature value",
        'FEATURE_VALUE_LOW': "Low",
        'FEATURE_VALUE_HIGH': "High",
        'JOINT_VALUE': "Joint SHAP value",
        'MODEL_OUTPUT': "Model output value"
    }
    
def __change_shap_base_value(base_value, new_base_value, shap_values) -> np.ndarray:
    """Shift SHAP base value to a new value. This function assumes that `base_value` and `new_base_value` are scalars
    and that `shap_values` is a two or three dimensional array.
    """
    # matrix of shap_values
    if shap_values.ndim == 2:
        return shap_values + (base_value - new_base_value) / shap_values.shape[1]

    # cube of shap_interaction_values
    main_effects = shap_values.shape[1]
    all_effects = main_effects * (main_effects + 1) // 2
    temp = (base_value - new_base_value) / all_effects / 2  # divided by 2 because interaction effects are halved
    shap_values = shap_values + temp
    # Add the other half to the main effects on the diagonal
    idx = np.diag_indices_from(shap_values[0])
    shap_values[:, idx[0], idx[1]] += temp
    return shap_values


def __decision_plot_matplotlib(
    base_value,
    cumsum,
    ascending,
    feature_display_count,
    features,
    feature_names,
    highlight,
    plot_color,
    axis_color,
    y_demarc_color,
    xlim,
    alpha,
    color_bar,
    auto_size_plot,
    title,
    show,
    legend_labels,
    legend_location,
    df, 
    color2, 
    cbar_label,
    cnorm,
    cmap,
    color_bar2
):
    """Matplotlib rendering for decision_plot()"""
    # image size
    row_height = 0.4
    if auto_size_plot:
        pl.gcf().set_size_inches(8, feature_display_count * row_height + 1.5)

    # draw vertical line indicating center
    pl.axvline(x=base_value, color="#999999", zorder=-1)

    # draw horizontal dashed lines for each feature contribution
    for i in range(1, feature_display_count):
        pl.axhline(y=i, color=y_demarc_color, lw=0.5, dashes=(1, 5), zorder=-1)

    # initialize highlighting
    linestyle = np.array("-", dtype=object)
    linestyle = np.repeat(linestyle, cumsum.shape[0])
    linewidth = np.repeat(1, cumsum.shape[0])
    if highlight is not None:
        linestyle[highlight] = "-."
        linewidth[highlight] = 2

    # plot each observation's cumulative SHAP values.
    ax = pl.gca()
    ax.set_xlim(xlim)

    if df is None:
        m = cm.ScalarMappable(cmap=plot_color)
        m.set_clim(xlim)
        y_pos = np.arange(0, feature_display_count + 1)
        lines = []

        for i in range(cumsum.shape[0]):
            o = pl.plot(
                cumsum[i, :],
                y_pos,
                color=m.to_rgba(cumsum[i, -1], alpha),
                linewidth=linewidth[i],
                linestyle=linestyle[i]
            )
            lines.append(o[0])
    else:
        m = cm.ScalarMappable(cmap=cmap,norm=cnorm)
        m.set_clim([df[color2].min(),df[color2].max()])
        y_pos = np.arange(0, feature_display_count + 1)
        lines = []
        for i in range(cumsum.shape[0]):
            o = pl.plot(
                cumsum[i, :],
                y_pos,
                color=m.to_rgba(df[color2].values[i], alpha),
                linewidth=linewidth[i],
                linestyle=linestyle[i]
            )
            lines.append(o[0])

    # determine font size. if ' *\n' character sequence is found (as in interaction labels), use a smaller
    # font. we don't shrink the font for all interaction plots because if an interaction term is not
    # in the display window there is no need to shrink the font.
    s = next((s for s in feature_names if " *\n" in s), None)
    fontsize = 13 if s is None else 9

    # if there is a single observation and feature values are supplied, print them.
    if (cumsum.shape[0] == 1) and (features is not None):
        renderer = pl.gcf().canvas.get_renderer()
        inverter = pl.gca().transData.inverted()
        y_pos = y_pos + 0.5
        for i in range(feature_display_count):
            v = features[0, i]
            if isinstance(v, str):
                v = f"({str(v).strip()})"
            else:
                v = "({})".format(f"{v:,.3f}".rstrip("0").rstrip("."))
            t = ax.text(np.max(cumsum[0, i:(i + 2)]), y_pos[i], "  " + v, fontsize=fontsize,
                    horizontalalignment="left", verticalalignment="center_baseline", color="#666666")
            bb = inverter.transform_bbox(t.get_window_extent(renderer=renderer))
            if bb.xmax > xlim[1]:
                t.set_text(v + "  ")
                t.set_x(np.min(cumsum[0, i:(i + 2)]))
                t.set_horizontalalignment("right")
                bb = inverter.transform_bbox(t.get_window_extent(renderer=renderer))
                if bb.xmin < xlim[0]:
                    t.set_text(v)
                    t.set_x(xlim[0])
                    t.set_horizontalalignment("left")

    # style axes
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labeltop=True)
    pl.yticks(np.arange(feature_display_count) + 0.5, feature_names, fontsize=fontsize)
    ax.tick_params("x", labelsize=11)
    pl.ylim(0, feature_display_count)
    pl.xlabel(labels["MODEL_OUTPUT"], fontsize=13)

    # draw the color bar - must come after axes styling
    if color_bar:
        m = cm.ScalarMappable(cmap=plot_color)
        m.set_array(np.array([0, 1]))

        # place the colorbar
        pl.ylim(0, feature_display_count + 0.25)
        ax_cb = ax.inset_axes([xlim[0], feature_display_count, xlim[1] - xlim[0], 0.25], transform=ax.transData)
        cb = pl.colorbar(m, ticks=[0, 1], orientation="horizontal", cax=ax_cb)
        cb.set_ticklabels([])
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(alpha)
        cb.outline.set_visible(False)

        # re-activate the main axis for drawing.
        pl.sca(ax)

    # draw the color bar for line color
    if color_bar2:
        m = cm.ScalarMappable(cmap=cmap, norm = cnorm)
        # m.set_array(np.array([0, 1]))

        # place the colorbar
        ax_cb = ax.inset_axes([xlim[1]+(xlim[1]-xlim[0])*.05, 
                                0,
                                (xlim[1]-xlim[0])*.05,
                                feature_display_count], 
                                transform=ax.transData)
        cb = pl.colorbar(m, orientation="vertical", cax=ax_cb, label = cbar_label)
        # cb.set_ticklabels([])
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(alpha)
        cb.outline.set_visible(False)

        # re-activate the main axis for drawing.
        pl.sca(ax)

    if title:
        # TODO decide on style/size
        pl.title(title)

    if ascending:
        pl.gca().invert_yaxis()

    if legend_labels is not None:
        ax.legend(handles=lines, labels=legend_labels, loc=legend_location)

    if show:
        pl.show()


class DecisionPlotResult:
    """The optional return value of decision_plot.

    The class attributes can be used to apply the same scale and feature ordering to other decision plots.
    """

    def __init__(self, base_value, shap_values, feature_names, feature_idx, xlim):
        """Example
        -------
        Plot two decision plots using the same feature order and x-axis.
        >>> range1, range2 = range(20), range(20, 40)
        >>> r = decision_plot(base, shap_values[range1], features[range1], return_objects=True)
        >>> decision_plot(base, shap_values[range2], features[range2], feature_order=r.feature_idx, xlim=r.xlim)

        Parameters
        ----------
        base_value : float
            The base value used in the plot. For multioutput models,
            this will be the mean of the base values. This will inherit `new_base_value` if specified.

        shap_values : numpy.ndarray
            The `shap_values` passed to decision_plot re-ordered based on `feature_order`. If SHAP interaction values
            are passed to decision_plot, `shap_values` is a 2D (matrix) representation of the interactions. See
            `feature_names` to locate the feature positions. If `new_base_value` is specified, the SHAP values are
            relative to the new base value.

        feature_names : list of str
            The feature names used in the plot in the order specified in the decision_plot parameter `feature_order`.

        feature_idx : numpy.ndarray
            The index used to order `shap_values` based on `feature_order`. This attribute can be used to specify
            identical feature ordering in multiple decision plots.

        xlim : tuple[float, float]
            The x-axis limits. This attributed can be used to specify the same x-axis in multiple decision plots.

        """
        self.base_value = base_value
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.feature_idx = feature_idx
        self.xlim = xlim

def createShapExplanation(shap_values, values = None, base_values = None, data = None):
    if values is None:
        values = shap_values.values
    if base_values is None:
        base_values = shap_values.base_values.squeeze()
    if data is None:
        data = shap_values.data
    display_data = shap_values.display_data
    instance_names = shap_values.instance_names
    feature_names = shap_values.feature_names
    output_names = shap_values.output_names
    output_indexes = shap_values.output_indexes
    lower_bounds = shap_values.lower_bounds
    upper_bounds = shap_values.upper_bounds
    error_std = shap_values.error_std
    main_effects = shap_values.main_effects
    hierarchical_values = shap_values.hierarchical_values
    clustering = shap_values.clustering
    new_exp = shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        display_data=display_data,
        instance_names=instance_names,
        feature_names=feature_names,
        output_indexes=output_indexes,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        error_std=error_std,
        main_effects=main_effects,
        hierarchical_values=hierarchical_values,
        clustering=clustering,
    )
    return new_exp

def decision(
    base_value,
    shap_values,
    features=None,
    feature_names=None,
    feature_order="importance",
    feature_display_range=None,
    highlight=None,
    link="identity",
    plot_color=None,
    axis_color="#333333",
    y_demarc_color="#333333",
    alpha=None,
    color_bar=True,
    auto_size_plot=True,
    title=None,
    xlim=None,
    show=True,
    return_objects=False,
    ignore_warnings=False,
    new_base_value=None,
    legend_labels=None,
    legend_location="best",
    df = None, 
    color2 = None, 
    cbar_label = None,
    cnorm = LogNorm(),
    cmap = 'RdBu',
    color_bar2 = True
) -> Union[DecisionPlotResult, None]:
    """Visualize model decisions using cumulative SHAP values.

    Each plotted line explains a single model prediction. If a single prediction is plotted, feature values will be
    printed in the plot (if supplied). If multiple predictions are plotted together, feature values will not be printed.
    Plotting too many predictions together will make the plot unintelligible.

    Parameters
    ----------
    base_value : float or numpy.ndarray
        This is the reference value that the feature contributions start from. Usually, this is
        ``explainer.expected_value``.

    shap_values : numpy.ndarray
        Matrix of SHAP values (# features) or (# samples x # features) from
        ``explainer.shap_values()``. Or cube of SHAP interaction values (# samples x
        # features x # features) from ``explainer.shap_interaction_values()``.

    features : numpy.array or pandas.Series or pandas.DataFrame or numpy.ndarray or list
        Matrix of feature values (# features) or (# samples x # features). This provides the values of all the
        features and, optionally, the feature names.

    feature_names : list or numpy.ndarray
        List of feature names (# features). If ``None``, names may be derived from the
        ``features`` argument if a Pandas object is provided. Otherwise, numeric feature
        names will be generated.

    feature_order : str or None or list or numpy.ndarray
        Any of "importance" (the default), "hclust" (hierarchical clustering), ``None``,
        or a list/array of indices.

    feature_display_range: slice or range
        The slice or range of features to plot after ordering features by ``feature_order``. A step of 1 or ``None``
        will display the features in ascending order. A step of -1 will display the features in descending order. If
        ``feature_display_range=None``, ``slice(-1, -21, -1)`` is used (i.e. show the last 20 features in descending order).
        If ``shap_values`` contains interaction values, the number of features is automatically expanded to include all
        possible interactions: N(N + 1)/2 where N = ``shap_values.shape[1]``.

    highlight : Any
        Specify which observations to draw in a different line style. All numpy indexing methods are supported. For
        example, list of integer indices, or a bool array.

    link : str
        Use "identity" or "logit" to specify the transformation used for the x-axis. The "logit" link transforms
        log-odds into probabilities.

    plot_color : str or matplotlib.colors.ColorMap
        Color spectrum used to draw the plot lines. If ``str``, a registered matplotlib color name is assumed.

    axis_color : str or int
        Color used to draw plot axes.

    y_demarc_color : str or int
        Color used to draw feature demarcation lines on the y-axis.

    alpha : float
        Alpha blending value in [0, 1] used to draw plot lines.

    color_bar : bool
        Whether to draw the color bar (legend).

    auto_size_plot : bool
        Whether to automatically size the matplotlib plot to fit the number of features
        displayed. If ``False``, specify the plot size using matplotlib before calling
        this function.

    title : str
        Title of the plot.

    xlim: tuple[float, float]
        The extents of the x-axis (e.g. ``(-1.0, 1.0)``). If not specified, the limits
        are determined by the maximum/minimum predictions centered around base_value
        when ``link="identity"``. When ``link="logit"``, the x-axis extents are ``(0,
        1)`` centered at 0.5. ``xlim`` values are not transformed by the ``link``
        function. This argument is provided to simplify producing multiple plots on the
        same scale for comparison.

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    return_objects : bool
        Whether to return a :obj:`DecisionPlotResult` object containing various plotting
        features. This can be used to generate multiple decision plots using the same
        feature ordering and scale.

    ignore_warnings : bool
        Plotting many data points or too many features at a time may be slow, or may create very large plots. Set
        this argument to ``True`` to override hard-coded limits that prevent plotting large amounts of data.

    new_base_value : float
        SHAP values are relative to a base value. By default, this base value is the
        expected value of the model's raw predictions. Use ``new_base_value`` to shift
        the base value to an arbitrary value (e.g. the cutoff point for a binary
        classification task).

    legend_labels : list of str
        List of legend labels. If ``None``, legend will not be shown.

    legend_location : str
        Legend location. Any of "best", "upper right", "upper left", "lower left", "lower right", "right",
        "center left", "center right", "lower center", "upper center", "center".

    df: DataFrame
        dataframe used to color lines, with the same number of rows with SHAP array

    color2: str
        column names in df used for coloring lines

    cbar_label: str
        colorbar label for line color plotting

    cnorm: maplotlib Normalize object
        norm for line color plotting

    cmap: str
        cmap for line color

    color_bar2: bool
        whether draw the color bar for line color or not

    Returns
    -------
    DecisionPlotResult or None
        Returns a :obj:`DecisionPlotResult` object if ``return_objects=True``. Returns ``None`` otherwise (the default).

    Examples
    --------
    Plot two decision plots using the same feature order and x-axis.

        >>> range1, range2 = range(20), range(20, 40)
        >>> r = decision_plot(base, shap_values[range1], features[range1], return_objects=True)
        >>> decision_plot(base, shap_values[range2], features[range2], feature_order=r.feature_idx, xlim=r.xlim)

    See more `decision plot examples here <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/decision_plot.html>`_.

    """
    # code taken from force_plot. auto unwrap the base_value
    if type(base_value) == np.ndarray and len(base_value) == 1:
        base_value = base_value[0]

    if isinstance(base_value, list) or isinstance(shap_values, list):
        raise TypeError("Looks like multi output. Try base_value[i] and shap_values[i], "
                        "or use shap.multioutput_decision_plot().")

    # validate shap_values
    if not isinstance(shap_values, np.ndarray):
        raise TypeError("The shap_values arg is the wrong type. Try explainer.shap_values().")

    # calculate the various dimensions involved (observations, features, interactions, display, etc.
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    observation_count = shap_values.shape[0]
    feature_count = shap_values.shape[1]

    # code taken from force_plot. convert features from other types.
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns.to_list()
        features = features.values
    elif isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = features.index.to_list()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and features.ndim == 1 and feature_names is None:
        feature_names = features.tolist()
        features = None

    # the above code converts features to either None or np.ndarray. if features is something else at this point,
    # there's a problem.
    if not isinstance(features, (np.ndarray, type(None))):
        raise TypeError("The features arg uses an unsupported type.")
    if (features is not None) and (features.ndim == 1):
        features = features.reshape(1, -1)

    # validate/generate feature_names. at this point, feature_names does not include interactions.
    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(feature_count)]
    elif len(feature_names) != feature_count:
        raise ValueError("The feature_names arg must include all features represented in shap_values.")
    elif not isinstance(feature_names, (list, np.ndarray)):
        raise TypeError("The feature_names arg requires a list or numpy array.")

    # transform interactions cube to a matrix and generate interaction names.
    if shap_values.ndim == 3:
        # flatten
        triu_count = feature_count * (feature_count - 1) // 2
        idx_diag = np.diag_indices_from(shap_values[0])
        idx_triu = np.triu_indices_from(shap_values[0], 1)
        a = np.ndarray((observation_count, feature_count + triu_count), shap_values.dtype)
        a[:, :feature_count] = shap_values[:, idx_diag[0], idx_diag[1]]
        a[:, feature_count:] = shap_values[:, idx_triu[0], idx_triu[1]] * 2
        shap_values = a
        # names
        a = [None] * shap_values.shape[1]
        a[:feature_count] = feature_names
        for i, row, col in zip(range(feature_count, shap_values.shape[1]), idx_triu[0], idx_triu[1]):
            a[i] = f"{feature_names[row]} *\n{feature_names[col]}"
        feature_names = a
        feature_count = shap_values.shape[1]
        features = None  # Can't use feature values for interactions...

    # determine feature order
    if isinstance(feature_order, list):
        feature_idx = np.array(feature_order)
    elif isinstance(feature_order, np.ndarray):
        feature_idx = feature_order
    elif (feature_order is None) or (feature_order.lower() == "none"):
        feature_idx = np.arange(feature_count)
    elif feature_order == "importance":
        feature_idx = np.argsort(np.sum(np.abs(shap_values), axis=0))
    elif feature_order == "hclust":
        feature_idx = np.array(hclust_ordering(shap_values.transpose()))
    else:
        raise ValueError("The feature_order arg requires 'importance', 'hclust', 'none', or an integer list/array "
                         "of feature indices.")

    if (feature_idx.shape != (feature_count, )) or (not np.issubdtype(feature_idx.dtype, np.integer)):
        raise ValueError("A list or array has been specified for the feature_order arg. The length must match the "
                         "feature count and the data type must be integer.")

    # validate and convert feature_display_range to a slice. prevents out of range errors later.
    if feature_display_range is None:
        feature_display_range = slice(-1, -21, -1)  # show last 20 features in descending order.
    elif not isinstance(feature_display_range, (slice, range)):
        raise TypeError("The feature_display_range arg requires a slice or a range.")
    elif feature_display_range.step not in (-1, 1, None):
        raise ValueError("The feature_display_range arg supports a step of 1, -1, or None.")
    elif isinstance(feature_display_range, range):
        # Negative values in a range are not the same as negs in a slice. Consider range(2, -1, -1) == [2, 1, 0],
        # but slice(2, -1, -1) == [] when len(features) > 2. However, range(2, -1, -1) == slice(2, -inf, -1) after
        # clipping.
        a = np.iinfo(np.integer).min
        feature_display_range = slice(
            feature_display_range.start if feature_display_range.start >= 0 else a,  # should never happen, but...
            feature_display_range.stop if feature_display_range.stop >= 0 else a,
            feature_display_range.step
        )

    # apply new_base_value
    if new_base_value is not None:
        shap_values = __change_shap_base_value(base_value, new_base_value, shap_values)
        base_value = new_base_value

    # use feature_display_range to determine which features will be plotted. convert feature_display_range to
    # ascending indices and expand by one in the negative direction. why? we are plotting the change in prediction
    # for every feature. this requires that we include the value previous to the first displayed feature
    # (i.e. i_0 - 1 to i_n).
    a = feature_display_range.indices(feature_count)
    ascending = True
    if a[2] == -1:  # The step
        ascending = False
        a = (a[1] + 1, a[0] + 1, 1)
    feature_display_count = a[1] - a[0]
    shap_values = shap_values[:, feature_idx]
    if a[0] == 0:
        cumsum = np.ndarray((observation_count, feature_display_count + 1), shap_values.dtype)
        cumsum[:, 0] = base_value
        cumsum[:, 1:] = base_value + np.nancumsum(shap_values[:, 0:a[1]], axis=1)
    else:
        cumsum = base_value + np.nancumsum(shap_values, axis=1)[:, (a[0] - 1):a[1]]

    # Select and sort feature names and features according to the range selected above
    feature_names = np.array(feature_names)
    feature_names_display = feature_names[feature_idx[a[0]:a[1]]].tolist()
    feature_names = feature_names[feature_idx].tolist()
    features_display = None if features is None else features[:, feature_idx[a[0]:a[1]]]

    # throw large data errors
    if not ignore_warnings:
        if observation_count > 2000:
            raise RuntimeError(f"Plotting {observation_count} observations may be slow. Consider subsampling or set "
                               "ignore_warnings=True to ignore this message.")
        if feature_display_count > 200:
            raise RuntimeError(f"Plotting {feature_display_count} features may create a very large plot. Set "
                               "ignore_warnings=True to ignore this "
                               "message.")
        if feature_count * observation_count > 100000000:
            raise RuntimeError(f"Processing SHAP values for {feature_count} features over {observation_count} observations may be slow. Set "
                               "ignore_warnings=True to ignore this "
                               "message.")

    # convert values based on link and update x-axis extents
    create_xlim = xlim is None
    link = convert_to_link(link)
    base_value_saved = base_value
    if isinstance(link, LogitLink):
        base_value = link.finv(base_value)
        cumsum = link.finv(cumsum)
        if create_xlim:
            # Expand [0, 1] limits a little for a visual margin
            xlim = (-0.02, 1.02)
    elif create_xlim:
        xmin = np.min((cumsum.min(), base_value))
        xmax = np.max((cumsum.max(), base_value))
        # create a symmetric axis around base_value
        a, b = (base_value - xmin), (xmax - base_value)
        if a > b:
            xlim = (base_value - a, base_value + a)
        else:
            xlim = (base_value - b, base_value + b)
        # Adjust xlim to include a little visual margin.
        a = (xlim[1] - xlim[0]) * 0.02
        xlim = (xlim[0] - a, xlim[1] + a)

    # Initialize style arguments
    if alpha is None:
        alpha = 1.0

    if plot_color is None:
        plot_color = red_blue

    __decision_plot_matplotlib(
        base_value,
        cumsum,
        ascending,
        feature_display_count,
        features_display,
        feature_names_display,
        highlight,
        plot_color,
        axis_color,
        y_demarc_color,
        xlim,
        alpha,
        color_bar,
        auto_size_plot,
        title,
        show,
        legend_labels,
        legend_location,
        df, 
        color2, 
        cbar_label,
        cnorm,
        cmap,
        color_bar2
    )

    if not return_objects:
        return None

    return DecisionPlotResult(base_value_saved, shap_values, feature_names, feature_idx, xlim)


def multioutput_decision(base_values, shap_values, row_index, **kwargs) -> Union[DecisionPlotResult, None]:

    """Decision plot for multioutput models.

    Plots all outputs for a single observation. By default, the plotted base value will be the mean of base_values
    unless new_base_value is specified. Supports both SHAP values and SHAP interaction values.

    Parameters
    ----------
    base_values : list of float
        This is the reference value that the feature contributions start from. Use explainer.expected_value.

    shap_values : list of numpy.ndarray
        A multioutput list of SHAP matrices or SHAP cubes from explainer.shap_values() or
        explainer.shap_interaction_values(), respectively.

    row_index : int
        The integer index of the row to plot.

    **kwargs : Any
        Arguments to be passed on to decision_plot().

    Returns
    -------
    DecisionPlotResult or None
        Returns a DecisionPlotResult object if `return_objects=True`. Returns `None` otherwise (the default).

    """
    if not (isinstance(base_values, list) and isinstance(shap_values, list)):
        raise ValueError("The base_values and shap_values args expect lists.")

    # convert arguments to arrays for simpler handling
    base_values = np.array(base_values)
    if not ((base_values.ndim == 1) or (np.issubdtype(base_values.dtype, np.number))):
        raise ValueError("The base_values arg should be a list of scalars.")
    shap_values = np.array(shap_values)
    if shap_values.ndim not in [3, 4]:
        raise ValueError("The shap_values arg should be a list of two or three dimensional SHAP arrays.")
    if shap_values.shape[0] != base_values.shape[0]:
        raise ValueError("The base_values output length is different than shap_values.")

    # shift shap base values to mean of base values
    base_values_mean = base_values.mean()
    for i in range(shap_values.shape[0]):
        shap_values[i] = __change_shap_base_value(base_values[i], base_values_mean, shap_values[i])

    # select the feature row corresponding to row_index
    if (kwargs is not None) and ("features" in kwargs):
        features = kwargs["features"]
        if isinstance(features, np.ndarray) and (features.ndim == 2):
            kwargs["features"] = features[[row_index]]
        elif isinstance(features, pd.DataFrame):
            kwargs["features"] = features.iloc[row_index]

    return decision(base_values_mean, shap_values[:, row_index, :], **kwargs)


def convert_ordering(ordering, shap_values):
    if issubclass(type(ordering), OpChain):
        ordering = ordering.apply(Explanation(shap_values))
    if issubclass(type(ordering), Explanation):
        if "argsort" in [op["name"] for op in ordering.op_history]:
            ordering = ordering.values
        else:
            ordering = ordering.argsort.flip.values
    return ordering

def heatmap(shap_values, instance_order=Explanation.hclust(), feature_values=Explanation.abs.mean(0), 
            feature_order=None, max_display=10, cmap=red_white_blue, show=True, plot_width=8, ax=None):
    """Create a heatmap plot of a set of SHAP values.

    This plot is designed to show the population substructure of a dataset using supervised
    clustering and a heatmap.
    Supervised clustering involves clustering data points not by their original
    feature values but by their explanations.
    By default, we cluster using :func:`shap.utils.hclust_ordering`,
    but any clustering can be used to order the samples.

    Parameters
    ----------
    shap_values : shap.Explanation
        A multi-row :class:`.Explanation` object that we want to visualize in a
        cluster ordering.

    instance_order : OpChain or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values and an axis, or
        a direct sample ordering given as an ``numpy.ndarray``.

    feature_values : OpChain or numpy.ndarray
            A function that returns a global summary value for each input feature, or an array of such values.

    feature_order : None, OpChain, or numpy.ndarray
        A function that returns a sort ordering given a matrix of SHAP values and an axis, or
        a direct input feature ordering given as an ``numpy.ndarray``.
        If ``None``, then we use ``feature_values.argsort``.

    max_display : int
        The maximum number of features to display (default is 10).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    plot_width : int, default 8
        The width of the heatmap plot.

    ax : matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax: matplotlib Axes
        Returns the Axes object with the plot drawn onto it.

    Examples
    --------
    See `heatmap plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/heatmap.html>`_.

    """

    # sort the SHAP values matrix by rows and columns
    values = shap_values.values
    if issubclass(type(feature_values), OpChain):
        feature_values = feature_values.apply(Explanation(values))
    if issubclass(type(feature_values), Explanation):
        feature_values = feature_values.values
    if feature_order is None:
        feature_order = np.argsort(-feature_values)
    elif issubclass(type(feature_order), OpChain):
        feature_order = feature_order.apply(Explanation(values))
    elif not hasattr(feature_order, "__len__"):
        raise Exception(f"Unsupported feature_order: {str(feature_order)}!")
    xlabel = "Instances"
    instance_order = convert_ordering(instance_order, shap_values)
    # if issubclass(type(instance_order), OpChain):
    #     #xlabel += " " + instance_order.summary_string("SHAP values")
    #     instance_order = instance_order.apply(Explanation(values))
    # elif not hasattr(instance_order, "__len__"):
    #     raise Exception("Unsupported instance_order: %s!" % str(instance_order))
    # else:
    #     instance_order_ops = None

    feature_names = np.array(shap_values.feature_names)[feature_order]
    values = shap_values.values[instance_order][:,feature_order]
    feature_values = feature_values[feature_order]

    # if we have more features than `max_display`, then group all the excess features
    # into a single feature
    if values.shape[1] > max_display:
        new_values = np.zeros((values.shape[0], max_display))
        new_values[:, :-1] = values[:, :max_display-1]
        new_values[:, -1] = values[:, max_display-1:].sum(1)
        new_feature_values = np.zeros(max_display)
        new_feature_values[:-1] = feature_values[:max_display-1]
        new_feature_values[-1] = feature_values[max_display-1:].sum()
        feature_names = [
            *feature_names[:max_display-1],
            f"Sum of {values.shape[1] - max_display + 1} other features",
        ]
        values = new_values
        feature_values = new_feature_values

    # define the plot size based on how many features we are plotting
    row_height = 0.5
    if ax is None:
        pl.gcf().set_size_inches(plot_width, values.shape[1] * row_height + 2.5)
        ax = pl.gca()

    # plot the matrix of SHAP values as a heat map
    vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
    ax.imshow(
        values.T,
        aspect=0.7 * values.shape[0] / values.shape[1],
        interpolation="nearest",
        vmin=min(vmin,-vmax),
        vmax=max(-vmin,vmax),
        cmap=cmap,
    )

    # adjust the axes ticks and spines for the heat map + f(x) line chart
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines[["left", "right"]].set_visible(True)
    ax.spines[["left", "right"]].set_bounds(values.shape[1] - row_height, -row_height)
    ax.spines[["top", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", direction="out")

    ax.set_ylim(values.shape[1] - row_height, -3)
    heatmap_yticks_pos = np.arange(values.shape[1])
    heatmap_yticks_labels = feature_names
    ax.yaxis.set_ticks(
        [-1.5, *heatmap_yticks_pos],
        [r"$f(x)$", *heatmap_yticks_labels],
        fontsize=13,
    )
    # remove the y-tick line for the f(x) label
    ax.yaxis.get_ticklines()[0].set_visible(False)

    ax.set_xlim(-0.5, values.shape[0] - 0.5)
    ax.set_xlabel(xlabel)

    # plot the f(x) line chart above the heat map
    ax.axhline(-1.5, color="#aaaaaa", linestyle="--", linewidth=0.5)
    fx = values.T.sum(0)
    ax.plot(
        -fx / np.abs(fx).max() - 1.5,
        color="#000000",
        linewidth=1,
    )

    # plot the bar plot on the right spine of the heat map
    bar_container = ax.barh(
        heatmap_yticks_pos,
        (feature_values / np.abs(feature_values).max()) * values.shape[0] / 20,
        height=0.7,
        align="center",
        color="#000000",
        left=values.shape[0] * 1.0 - 0.5,
        # color=[colors.red_rgb if shap_values[feature_inds[i]] > 0 else colors.blue_rgb for i in range(len(y_pos))]
    )
    for b in bar_container:
        b.set_clip_on(False)

    # draw the color bar
    import matplotlib.cm as cm
    m = cm.ScalarMappable(cmap=cmap)
    m.set_array([min(vmin, -vmax), max(-vmin, vmax)])
    cax = ax.inset_axes([1.1, -.05, .01, .98])
    cb = pl.colorbar(
        m,
        ticks=[min(vmin, -vmax), max(-vmin, vmax)],
        cax = cax,
        # aspect=80,
        # fraction=0.01,
        # pad=0.10,  # padding between the cb and the main axes
    )
    cb.set_label(labels["VALUE"], size=12, labelpad=-10)
    cb.ax.tick_params(labelsize=11, length=0)
    cb.set_alpha(1)
    cb.outline.set_visible(False)
    # bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
    # cb.ax.set_aspect((bbox.height - 0.9) * 15)
    # cb.draw_all()

    if show:
        pl.show()

    return ax

# TODO: Make the color bar a one-sided beeswarm plot so we can see the density along the color axis
def scatter(shap_values, 
            color="#1E88E5", 
            hist=True, 
            axis_color="#333333", 
            cmap=red_blue,
            dot_size=16, 
            x_jitter="auto", 
            alpha=1, 
            title=None, 
            xmin=None, 
            xmax=None, 
            ymin=None, 
            ymax=None,
            overlay=None, 
            ax=None, 
            ylabel="SHAP value", 
            show=True,
            df = None, 
            color2 = None, 
            cbar_label = None,
            cnorm = LogNorm(),
            fontsize = 10):
    """Create a SHAP dependence scatter plot, colored by an interaction feature.

    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extension of classical partial dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.

    Note that if you want to change the data being displayed, you can update the
    ``shap_values.display_features`` attribute and it will then be used for plotting instead of
    ``shap_values.data``.

    Parameters
    ----------
    shap_values : shap.Explanation
        A single column of an :class:`.Explanation` object (i.e.
        ``shap_values[:,"Feature A"]``).

    color : string or shap.Explanation
        How to color the scatter plot points. This can be a fixed color string, or an
        :class:`.Explanation` object. If it is an :class:`.Explanation` object, then the
        scatter plot points are colored by the feature that seems to have the strongest
        interaction effect with the feature given by the ``shap_values`` argument. This
        is calculated using :func:`shap.utils.approximate_interactions`. If only a
        single column of an :class:`.Explanation` object is passed, then that
        feature column will be used to color the data points.

    hist : bool
        Whether to show a light histogram along the x-axis to show the density of the
        data. Note that the histogram is normalized such that if all the points were in
        a single bin, then that bin would span the full height of the plot. Defaults to
        ``True``.

    x_jitter : 'auto' or float
        Adds random jitter to feature values by specifying a float between 0 to 1. May
        increase plot readability when a feature is discrete. By default, ``x_jitter``
        is chosen based on auto-detection of categorical features.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to
        show the density of the data points when using a large dataset.

    xmin : float or string
        Represents the lower bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    xmax : float or string
        Represents the upper bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    ax : matplotlib Axes object
        Optionally specify an existing matplotlib ``Axes`` object, into which the plot will be placed.
        In this case, we do not create a ``Figure``, otherwise we do.

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot
        to be customized further after it has been created.

    Examples
    --------
    See `scatter plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/scatter.html>`_.

    """
    labels = {
        'MAIN_EFFECT': "SHAP main effect value for\n%s",
        'INTERACTION_VALUE': "SHAP interaction value",
        'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
        'VALUE': "SHAP value (impact on model output)",
        'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
        'VALUE_FOR': "SHAP value for\n%s",
        'PLOT_FOR': "SHAP plot for %s",
        'FEATURE': "Feature %s",
        'FEATURE_VALUE': "Feature value",
        'FEATURE_VALUE_LOW': "Low",
        'FEATURE_VALUE_HIGH': "High",
        'JOINT_VALUE': "Joint SHAP value",
        'MODEL_OUTPUT': "Model output value"
    }

    assert str(type(shap_values)).endswith("Explanation'>"), "The shap_values parameter must be a shap.Explanation object!"

    # see if we are plotting multiple columns
    if not isinstance(shap_values.feature_names, str) and len(shap_values.feature_names) > 0:
        inds = np.argsort(np.abs(shap_values.values).mean(0))
        nan_min = np.nanmin(shap_values.values)
        nan_max = np.nanmax(shap_values.values)
        if ymin is None:
            ymin = nan_min - (nan_max - nan_min)/20
        if ymax is None:
            ymax = nan_max + (nan_max - nan_min)/20
        _ = pl.subplots(1, len(inds), figsize=(min(6 * len(inds), 15), 5))
        for i in inds:
            ax = pl.subplot(1,len(inds),i+1)
            scatter(shap_values[:,i], color=color, show=False, ax=ax, ymin=ymin, ymax=ymax)
            if overlay is not None:
                line_styles = ["solid", "dotted", "dashed"]
                for j, name in enumerate(overlay):
                    vals = overlay[name]
                    if isinstance(vals[i][0][0], (float, int)):
                        pl.plot(vals[i][0], vals[i][1], color="#000000", linestyle=line_styles[j], label=name)
            if i == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
                ax.set_yticks([])
                ax.spines['left'].set_visible(False)
        if overlay is not None:
            pl.legend()
        if show:
            pl.show()
        return

    if len(shap_values.shape) != 1:
        raise Exception("The passed Explanation object has multiple columns, please pass a single feature column to " + \
                        "shap.plots.dependence like: shap_values[:,column]")

    # this unpacks the explanation object for the code that was written earlier
    feature_names = [shap_values.feature_names]
    ind = 0
    shap_values_arr = shap_values.values.reshape(-1, 1)
    features = shap_values.data.reshape(-1, 1)
    if shap_values.display_data is None:
        display_features = features
    else:
        display_features = shap_values.display_data.reshape(-1, 1)
    interaction_index = None

    # unwrap explanation objects used for bounds
    if issubclass(type(xmin), Explanation):
        xmin = xmin.data
    if issubclass(type(xmax), Explanation):
        xmax = xmax.data
    if issubclass(type(ymin), Explanation):
        ymin = ymin.values
    if issubclass(type(ymax), Explanation):
        ymax = ymax.values

    # wrap np.arrays as Explanations
    if isinstance(color, np.ndarray):
        color = Explanation(values=color, base_values=None, data=color)

    # TODO: This stacking could be avoided if we use the new shap.utils.potential_interactions function
    if str(type(color)).endswith("Explanation'>"):
        shap_values2 = color
        if issubclass(type(shap_values2.feature_names), (str, int)):
            feature_names.append(shap_values2.feature_names)
            shap_values_arr = np.hstack([shap_values_arr, shap_values2.values.reshape(-1, len(feature_names)-1)])
            features = np.hstack([features, shap_values2.data.reshape(-1, len(feature_names)-1)])
            if shap_values2.display_data is None:
                display_features = np.hstack([display_features, shap_values2.data.reshape(-1, len(feature_names)-1)])
            else:
                display_features = np.hstack([display_features, shap_values2.display_data.reshape(-1, len(feature_names)-1)])
        else:
            feature_names2 = np.array(shap_values2.feature_names)
            mask = ~(feature_names[0] == feature_names2)
            feature_names.extend(feature_names2[mask])
            shap_values_arr = np.hstack([shap_values_arr, shap_values2.values[:,mask]])
            features = np.hstack([features, shap_values2.data[:,mask]])
            if shap_values2.display_data is None:
                display_features = np.hstack([display_features, shap_values2.data[:,mask]])
            else:
                display_features = np.hstack([display_features, shap_values2.display_data[:,mask]])
        color = None
        interaction_index = "auto"

    if isinstance(shap_values_arr, list):
        raise TypeError("The passed shap_values_arr are a list not an array! If you have a list of explanations try " \
                        "passing shap_values_arr[0] instead to explain the first output class of a multi-output model.")

    # convert from DataFrames if we got any
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values_arr.shape[1])]

    # allow vectors to be passed
    if len(shap_values_arr.shape) == 1:
        shap_values_arr = np.reshape(shap_values_arr, (len(shap_values_arr), 1))
    if len(features.shape) == 1:
        features = np.reshape(features, (len(features), 1))

    ind = convert_name(ind, shap_values_arr, feature_names)

    # pick jitter for categorical features
    vals = np.sort(np.unique(features[:,ind]))
    min_dist = np.inf
    for i in range(1,len(vals)):
        d = vals[i] - vals[i-1]
        if d > 1e-8 and d < min_dist:
            min_dist = d
    num_points_per_value = len(features[:,ind]) / len(vals)
    if num_points_per_value < 10:
        #categorical = False
        if x_jitter == "auto":
            x_jitter = 0
    elif num_points_per_value < 100:
        #categorical = True
        if x_jitter == "auto":
            x_jitter = min_dist * 0.1
    else:
        #categorical = True
        if x_jitter == "auto":
            x_jitter = min_dist * 0.2

    # guess what other feature as the stongest interaction with the plotted feature
    if not hasattr(ind, "__len__"):
        if interaction_index == "auto":
            interaction_index = approximate_interactions(ind, shap_values_arr, features)[0]
        interaction_index = convert_name(interaction_index, shap_values_arr, feature_names)
    categorical_interaction = False

    # create a matplotlib figure, if `ax` hasn't been specified.
    if not ax:
        figsize = (7.5, 5) if interaction_index != ind and interaction_index is not None else (6, 5)
        fig = pl.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # plotting SHAP interaction values
    if len(shap_values_arr.shape) == 3 and hasattr(ind, "__len__") and len(ind) == 2:
        ind1 = convert_name(ind[0], shap_values_arr, feature_names)
        ind2 = convert_name(ind[1], shap_values_arr, feature_names)
        if ind1 == ind2:
            proj_shap_values_arr = shap_values_arr[:, ind2, :]
        else:
            proj_shap_values_arr = shap_values_arr[:, ind2, :] * 2  # off-diag values are split in half

        # there is no interaction coloring for the main effect
        if ind1 == ind2:
            fig.set_size_inches(6, 5, forward=True)

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_legacy(
            ind1, proj_shap_values_arr, features, feature_names=feature_names,
            interaction_index=(None if ind1 == ind2 else ind2), display_features=display_features, ax=ax, show=False,
            xmin=xmin, xmax=xmax, x_jitter=x_jitter, alpha=alpha
        )
        if ind1 == ind2:
            ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            pl.show()
        return

    assert shap_values_arr.shape[0] == features.shape[0], \
        "'shap_values_arr' and 'features' values must have the same number of rows!"
    assert shap_values_arr.shape[1] == features.shape[1], \
        "'shap_values_arr' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    oinds = np.arange(shap_values_arr.shape[0]) # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)
    xv = encode_array_if_needed(features[oinds, ind])
    xd = display_features[oinds, ind]

    s = shap_values_arr[oinds, ind]
    if isinstance(xd[0], str):
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    name = feature_names[ind]

    # get both the raw and display color values
    color_norm = None
    if interaction_index is not None:
        interaction_feature_values = encode_array_if_needed(features[:, interaction_index])
        cv = interaction_feature_values
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(cv.astype(float), 5)
        chigh = np.nanpercentile(cv.astype(float), 95)
        if clow == chigh:
            clow = np.nanmin(cv.astype(float))
            chigh = np.nanmax(cv.astype(float))
        if isinstance(cd[0], str):
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        # discritize colors for categorical features
        if categorical_interaction and clow != chigh:
            clow = np.nanmin(cv.astype(float))
            chigh = np.nanmax(cv.astype(float))
            bounds = np.linspace(clow, chigh, min(int(chigh - clow + 2), cmap.N-1))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N-1)

    # optionally add jitter to feature values
    xv_no_jitter = xv.copy()
    if x_jitter > 0:
        if x_jitter > 1:
            x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals) # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.random_sample(size = len(xv))*jitter_amount) - (jitter_amount/2)


    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:

        # plot the nan values in the interaction feature as grey
        cvals = features[oinds, interaction_index].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        if color_norm is None:
            vmin = clow
            vmax = chigh
        else:
            vmin = vmax = None
        ax.axhline(0, color="#888888", lw=0.5, dashes=(1, 5), zorder=3)
        if df is not None and color2 is not None:                            # using colors in df as the color of scatter plot

            cvals = df[color2].values
            vmin = df[color2].min()
            vmax = df[color2].max()
            p = ax.scatter(
                xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
                cmap=cmap, alpha=alpha, norm = cnorm,
                rasterized=len(xv) > 500
            )
            p.set_array(cvals[xv_notnan])

        else:
            p = ax.scatter(
                xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
                cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax,
                norm=color_norm, rasterized=len(xv) > 500
            )
            p.set_array(cvals[xv_notnan])
    else:
        if df is not None and color2 is not None:                            # using colors in df as the color of scatter plot
            cvals = df[color2].values
            vmin = df[color2].min()
            vmax = df[color2].max()
            p = ax.scatter(
                xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
                cmap=cmap, alpha=alpha, norm = cnorm,
                rasterized=len(xv) > 500
            )
            p.set_array(cvals[xv_notnan])
        else:
            p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
                        alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        cax = ax.inset_axes([1.02, 0, .03, 1])
        if isinstance(cd[0], str):
            tick_positions = np.array([cname_map[n] for n in cnames])
            tick_positions *= 1 - 1 / len(cnames)
            tick_positions += 0.5 * (chigh - clow) / (chigh - clow + 1)
            cb = pl.colorbar(p, ticks=tick_positions, cax = cax)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar(p, cax = cax)
        if df is not None and color2 is not None: 
            cb.set_label(cbar_label, size = fontsize)
        else:
            cb.set_label(feature_names[interaction_index], size=fontsize)
        cb.ax.tick_params(labelsize=fontsize-1)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
#         bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#         cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if isinstance(xmin, str) and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if isinstance(xmax, str) and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

        ax.set_xlim(xmin, xmax)

    if ymin is not None or ymax is not None:
        # if type(ymin) == str and ymin.startswith("percentile"):
        #     ymin = np.nanpercentile(xv, float(ymin[11:-1]))
        # if type(ymax) == str and ymax.startswith("percentile"):
        #     ymax = np.nanpercentile(xv, float(ymax[11:-1]))

        if ymin is None or ymin == np.nanmin(xv):
            ymin = np.nanmin(xv) - (ymax - np.nanmin(xv))/20
        if ymax is None or ymax == np.nanmax(xv):
            ymax = np.nanmax(xv) + (np.nanmax(xv) - ymin)/20

        ax.set_ylim(ymin, ymax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # the histogram of the data
    if hist:
        ax2 = ax.twinx()
        #n, bins, patches =
        xlim = ax.get_xlim()
        xvals = np.unique(xv_no_jitter)

        if len(xvals) / len(xv_no_jitter) < 0.2 and len(xvals) < 75 and np.max(xvals) < 75 and np.min(xvals) >= 0:
            np.sort(xvals)
            bin_edges = []
            for i in range(int(np.max(xvals)+1)):
                bin_edges.append(i-0.5)

                #bin_edges.append((xvals[i] + xvals[i+1])/2)
            bin_edges.append(int(np.max(xvals))+0.5)

            lim = np.floor(np.min(xvals) - 0.5) + 0.5, np.ceil(np.max(xvals) + 0.5) - 0.5
            ax.set_xlim(lim)
        else:
            if len(xv_no_jitter) >= 500:
                bin_edges = 50
            elif len(xv_no_jitter) >= 200:
                bin_edges = 20
            elif len(xv_no_jitter) >= 100:
                bin_edges = 10
            else:
                bin_edges = 5

        ax2.hist(xv[~np.isnan(xv)], bin_edges, density=False, facecolor='#000000', alpha=0.1, range=(xlim[0], xlim[1]), zorder=-1)
        ax2.set_ylim(0,len(xv))

        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        ax2.yaxis.set_ticks([])
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

    pl.sca(ax)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=fontsize)
    ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=fontsize)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=fontsize-1)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if isinstance(xd[0], str):
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, fontdict=dict(rotation='vertical', fontsize=fontsize-1))
    if show:
        with warnings.catch_warnings(): # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            pl.show()
    
    return ax


def dependence_legacy(ind, shap_values=None, features=None, feature_names=None, display_features=None,
                      interaction_index="auto",
                      color="#1E88E5", axis_color="#333333", cmap=None,
                      dot_size=16, x_jitter=0, alpha=1, title=None, xmin=None, xmax=None, ax=None, show=True,
                      ymin=None, ymax=None):
    """Create a SHAP dependence plot, colored by an interaction feature.

    Plots the value of the feature on the x-axis and the SHAP value of the same feature
    on the y-axis. This shows how the model depends on the given feature, and is like a
    richer extension of the classical partial dependence plots. Vertical dispersion of the
    data points represents interaction effects. Grey ticks along the y-axis are data
    points where the feature's value was NaN.


    Parameters
    ----------
    ind : int or string
        If this is an int it is the index of the feature to plot. If this is a string it is
        either the name of the feature to plot, or it can have the form "rank(int)" to specify
        the feature with that rank (ordered by mean absolute SHAP value over all the samples).

    shap_values : numpy.array
        Matrix of SHAP values (# samples x # features).

    features : numpy.array or pandas.DataFrame
        Matrix of feature values (# samples x # features).

    feature_names : list
        Names of the features (length # features).

    display_features : numpy.array or pandas.DataFrame
        Matrix of feature values for visual display (such as strings instead of coded values).

    interaction_index : "auto", None, int, or string
        The index of the feature used to color the plot. The name of a feature can also be passed
        as a string. If "auto" then shap.common.approximate_interactions is used to pick what
        seems to be the strongest interaction (note that to find to true stongest interaction you
        need to compute the SHAP interaction values).

    x_jitter : float (0 - 1)
        Adds random jitter to feature values. May increase plot readability when feature
        is discrete.

    alpha : float
        The transparency of the data points (between 0 and 1). This can be useful to the
        show density of the data points when using a large dataset.

    xmin : float or string
        Represents the lower bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    xmax : float or string
        Represents the upper bound of the plot's x-axis. It can be a string of the format
        "percentile(float)" to denote that percentile of the feature's value used on the x-axis.

    ax : matplotlib Axes object
         Optionally specify an existing matplotlib Axes object, into which the plot will be placed.
         In this case we do not create a Figure, otherwise we do.

    ymin : float
        Represents the lower bound of the plot's y-axis.

    ymax : float
        Represents the upper bound of the plot's y-axis.

    """
    if cmap is None:
        cmap = colors.red_blue

    if isinstance(shap_values, list):
        raise TypeError("The passed shap_values are a list not an array! If you have a list of explanations try " \
                        "passing shap_values[0] instead to explain the first output class of a multi-output model.")

    # convert from DataFrames if we got any
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    if isinstance(display_features, pd.DataFrame):
        if feature_names is None:
            feature_names = display_features.columns
        display_features = display_features.values
    elif display_features is None:
        display_features = features

    if feature_names is None:
        feature_names = [labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]

    # allow vectors to be passed
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, (len(shap_values), 1))
    if len(features.shape) == 1:
        features = np.reshape(features, (len(features), 1))

    ind = convert_name(ind, shap_values, feature_names)

    # guess what other feature as the stongest interaction with the plotted feature
    if not hasattr(ind, "__len__"):
        if interaction_index == "auto":
            interaction_index = approximate_interactions(ind, shap_values, features)[0]
        interaction_index = convert_name(interaction_index, shap_values, feature_names)
    categorical_interaction = False

    # create a matplotlib figure, if `ax` hasn't been specified.
    if not ax:
        figsize = (7.5, 5) if interaction_index != ind and interaction_index is not None else (6, 5)
        fig = pl.figure(figsize=figsize)
        ax = fig.gca()
    else:
        fig = ax.get_figure()

    # plotting SHAP interaction values
    if len(shap_values.shape) == 3 and hasattr(ind, "__len__") and len(ind) == 2:
        ind1 = convert_name(ind[0], shap_values, feature_names)
        ind2 = convert_name(ind[1], shap_values, feature_names)
        if ind1 == ind2:
            proj_shap_values = shap_values[:, ind2, :]
        else:
            proj_shap_values = shap_values[:, ind2, :] * 2  # off-diag values are split in half

        # there is no interaction coloring for the main effect
        if ind1 == ind2:
            fig.set_size_inches(6, 5, forward=True)

        # TODO: remove recursion; generally the functions should be shorter for more maintainable code
        dependence_legacy(
            ind1, proj_shap_values, features, feature_names=feature_names,
            interaction_index=(None if ind1 == ind2 else ind2), display_features=display_features, ax=ax, show=False,
            xmin=xmin, xmax=xmax, x_jitter=x_jitter, alpha=alpha
        )
        if ind1 == ind2:
            ax.set_ylabel(labels['MAIN_EFFECT'] % feature_names[ind1])
        else:
            ax.set_ylabel(labels['INTERACTION_EFFECT'] % (feature_names[ind1], feature_names[ind2]))

        if show:
            pl.show()
        return

    assert shap_values.shape[0] == features.shape[0], \
        "'shap_values' and 'features' values must have the same number of rows!"
    assert shap_values.shape[1] == features.shape[1], \
        "'shap_values' must have the same number of columns as 'features'!"

    # get both the raw and display feature values
    oinds = np.arange(shap_values.shape[0]) # we randomize the ordering so plotting overlaps are not related to data ordering
    np.random.shuffle(oinds)

    xv = encode_array_if_needed(features[oinds, ind])

    xd = display_features[oinds, ind]
    s = shap_values[oinds, ind]
    if isinstance(xd[0], str):
        name_map = {}
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    # allow a single feature name to be passed alone
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    name = feature_names[ind]

    # get both the raw and display color values
    color_norm = None
    if interaction_index is not None:
        interaction_feature_values = encode_array_if_needed(features[:, interaction_index])
        cv = interaction_feature_values
        cd = display_features[:, interaction_index]
        clow = np.nanpercentile(cv.astype(float), 5)
        chigh = np.nanpercentile(cv.astype(float), 95)
        if clow == chigh:
            clow = np.nanmin(cv.astype(float))
            chigh = np.nanmax(cv.astype(float))
        if isinstance(cd[0], str):
            cname_map = {}
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        # discritize colors for categorical features
        if categorical_interaction and clow != chigh:
            clow = np.nanmin(cv.astype(float))
            chigh = np.nanmax(cv.astype(float))
            bounds = np.linspace(clow, chigh, min(int(chigh - clow + 2), cmap.N-1))
            color_norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N-1)

    # optionally add jitter to feature values
    if x_jitter > 0:
        if x_jitter > 1:
            x_jitter = 1
        xvals = xv.copy()
        if isinstance(xvals[0], float):
            xvals = xvals.astype(float)
            xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals) # returns a sorted array
        if len(xvals) >= 2:
            smallest_diff = np.min(np.diff(xvals))
            jitter_amount = x_jitter * smallest_diff
            xv += (np.random.random_sample(size = len(xv))*jitter_amount) - (jitter_amount/2)

    # the actual scatter plot, TODO: adapt the dot_size to the number of data points?
    xv_nan = np.isnan(xv)
    xv_notnan = np.invert(xv_nan)
    if interaction_index is not None:

        # plot the nan values in the interaction feature as grey
        cvals = interaction_feature_values[oinds].astype(np.float64)
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        p = ax.scatter(
            xv[xv_notnan], s[xv_notnan], s=dot_size, linewidth=0, c=cvals[xv_notnan],
            cmap=cmap, alpha=alpha,
            norm=color_norm, rasterized=len(xv) > 500
        )
        p.set_array(cvals[xv_notnan])
    else:
        p = ax.scatter(xv, s, s=dot_size, linewidth=0, color=color,
                       alpha=alpha, rasterized=len(xv) > 500)

    if interaction_index != ind and interaction_index is not None:
        # draw the color bar
        if isinstance(cd[0], str):
            tick_positions = [cname_map[n] for n in cnames]
            if len(tick_positions) == 2:
                tick_positions[0] -= 0.25
                tick_positions[1] += 0.25
            cb = pl.colorbar(p, ticks=tick_positions, ax=ax, aspect=80)
            cb.set_ticklabels(cnames)
        else:
            cb = pl.colorbar(p, ax=ax, aspect=80)

        cb.set_label(feature_names[interaction_index], size=13)
        cb.ax.tick_params(labelsize=11)
        if categorical_interaction:
            cb.ax.tick_params(length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
#         bbox = cb.ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#         cb.ax.set_aspect((bbox.height - 0.7) * 20)

    # handles any setting of xmax and xmin
    # note that we handle None,float, or "percentile(float)" formats
    if xmin is not None or xmax is not None:
        if isinstance(xmin, str) and xmin.startswith("percentile"):
            xmin = np.nanpercentile(xv, float(xmin[11:-1]))
        if isinstance(xmax, str) and xmax.startswith("percentile"):
            xmax = np.nanpercentile(xv, float(xmax[11:-1]))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = np.nanmin(xv) - (xmax - np.nanmin(xv))/20
        if xmax is None or xmax == np.nanmax(xv):
            xmax = np.nanmax(xv) + (np.nanmax(xv) - xmin)/20

        ax.set_xlim(xmin, xmax)

    # plot any nan feature values as tick marks along the y-axis
    xlim = ax.get_xlim()
    if interaction_index is not None:
        p = ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, c=cvals_imp[xv_nan], cmap=cmap, alpha=alpha,
            vmin=clow, vmax=chigh
        )
        p.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(xv_nan.sum()), s[xv_nan], marker=1,
            linewidth=2, color=color, alpha=alpha
        )
    ax.set_xlim(xlim)

    # make the plot more readable
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(labels['VALUE_FOR'] % name, color=axis_color, fontsize=13)

    if (ymin is not None) or (ymax is not None):
        if ymin is None:
            ymin = -ymax
        if ymax is None:
            ymax = -ymin

        ax.set_ylim(ymin, ymax)

    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if isinstance(xd[0], str):
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, fontdict=dict(rotation='vertical', fontsize=11))
    if show:
        with warnings.catch_warnings(): # ignore expected matplotlib warnings
            warnings.simplefilter("ignore", RuntimeWarning)
            pl.show()