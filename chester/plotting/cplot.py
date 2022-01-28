import os
import sys
import copy
import json
import argparse
import itertools
import numpy as np

# Matplotlib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rc('font', size=25)
matplotlib.rcParams['pdf.fonttype'] = 42  # Default type3 cannot be rendered in some templates
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['grid.alpha'] = 0.3
matplotlib.rcParams['axes.titlesize'] = 25
import matplotlib.ticker as tick

# rllab
sys.path.append('.')
from rllab.misc.ext import flatten
from rllab.viskit import core


# from rllab.misc import ext

# plotly
# import plotly.offline as po
# import plotly.graph_objs as go


def reload_data(data_paths):
    """
    Iterate through the data folder and organize each experiment into a list, with their progress data, hyper-parameters
    and also analyze all the curves and give the distinct hyper-parameters.
    :param data_path: Path of the folder storing all the data
    :return [exps_data, plottable_keys, distinct_params]
        exps_data: A list of the progress data for each curve. Each curve is an AttrDict with the key
                    'progress': A dictionary of plottable keys. The val of each key is an ndarray representing the
                        values of the key during training, or one column in the progress.txt file.
                    'params'/'flat_params': A dictionary of all hyperparameters recorded in 'variants.json' file.
        plottable_keys: A list of strings representing all the keys that can be plotted.
        distinct_params: A list of hyper-parameters which have different values among all the curves. This can be used
                    to split the graph into multiple figures. Each element is a tuple (param, list_of_values_to_take).
    """

    exps_data = copy.copy(core.load_exps_data(data_paths, disable_variant=False, ignore_missing_keys=True))
    plottable_keys = copy.copy(sorted(list(set(flatten(list(exp.progress.keys()) for exp in exps_data)))))
    distinct_params = copy.copy(sorted(core.extract_distinct_params(exps_data)))

    return exps_data, plottable_keys, distinct_params


def get_shaded_curve(selector, key, shade_type='variance'):
    """
    :param selector: Selector for a group of curves
    :param shade_type: Should be either 'variance' or 'median', indicating how the shades are calculated.
    :return: [y, y_lower, y_upper], representing the mean, upper and lower boundary of the shaded region
    """

    # First, get the progresses
    progresses = [exp.progress.get(key, np.array([np.nan])) for exp in selector.extract()]
    max_size = max(len(x) for x in progresses)
    progresses = [np.concatenate([ps, np.ones(max_size - len(ps)) * np.nan]) for ps in progresses]

    # Second, calculate the shaded area
    if shade_type == 'median':
        percentile25 = np.nanpercentile(
            progresses, q=25, axis=0)
        percentile50 = np.nanpercentile(
            progresses, q=50, axis=0)
        percentile75 = np.nanpercentile(
            progresses, q=75, axis=0)

        y = list(percentile50)
        y_upper = list(percentile75)
        y_lower = list(percentile25)
    elif shade_type == 'variance':
        means = np.nanmean(progresses, axis=0)
        stds = np.nanstd(progresses, axis=0)

        y = list(means)
        y_upper = list(means + stds)
        y_lower = list(means - stds)
    else:
        raise NotImplementedError

    return y, y_lower, y_upper


def get_group_selectors(exps, custom_series_splitter):
    """

    :param exps:
    :param custom_series_splitter:
    :return: A dictionary with (splitted_keys, group_selectors). Group selectors can be used to extract progresses.
    """
    splitted_dict = dict()
    for exp in exps:
        # Group exps by their series_splitter key
        # splitted_dict: {key:[exp1, exp2, ...]}
        key = custom_series_splitter(exp)
        if key not in splitted_dict:
            splitted_dict[key] = list()
        splitted_dict[key].append(exp)

    splitted = list(splitted_dict.items())
    # Group selectors: All the exps in one of the keys/legends
    # Group legends: All the different legends
    group_selectors = [core.Selector(list(x[1])) for x in splitted]
    group_legends = [x[0] for x in splitted]
    all_tuples = sorted(list(zip(group_selectors, group_legends)), key=lambda x: x[1], reverse=True)
    group_selectors = [x[0] for x in all_tuples]
    group_legends = [x[1] for x in all_tuples]
    return group_selectors, group_legends


def filter_save_name(save_name):
    save_name = save_name.replace('/', '_')
    save_name = save_name.replace('[', '_')
    save_name = save_name.replace(']', '_')
    save_name = save_name.replace(',', '_')
    save_name = save_name.replace(' ', '_')
    save_name = save_name.replace('0.', '0_')

    return save_name


def sliding_mean(data_array, window=5):
    data_array = np.array(data_array)
    new_list = []
    for i in range(len(data_array)):
        indices = list(range(max(i - window + 1, 0),
                             min(i + window + 1, len(data_array))))
        avg = 0
        for j in indices:
            avg += data_array[j]
        avg /= float(len(indices))
        new_list.append(avg)

    return np.array(new_list)


if __name__ == '__main__':
    data_path = '/Users/Dora/Projects/baselines_hrl/data/seuss/visual_rss_RopeFloat_0407'
    exps_data, plottable_keys, distinct_params = reload_data(data_path)

    # Example of extracting a single curve
    selector = core.Selector(exps_data)
    selector = selector.where('her_replay_strategy', 'balance_filter')
    y, y_lower, y_upper = get_shaded_curve(selector, 'test/success_state')
    _, ax = plt.subplots()

    color = core.color_defaults[0]
    ax.fill_between(range(len(y)), y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0, alpha=0.2)
    ax.plot(range(len(y)), y, color=color, label=plt.legend, linewidth=2.0)


    # Example of extracting all the curves
    def custom_series_splitter(x):
        params = x['flat_params']
        if 'use_ae_reward' in params and params['use_ae_reward']:
            return 'Auto Encoder'
        if params['her_replay_strategy'] == 'balance_filter':
            return 'Indicator+Balance+Filter'
        if params['env_kwargs.use_true_reward']:
            return 'Oracle'
        return 'Indicator'


    fig, ax = plt.subplots(figsize=(8, 5))

    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
        color = core.color_defaults[idx]

        y, y_lower, y_upper = get_shaded_curve(selector, 'test/success_state')

        ax.plot(range(len(y)), y, color=color, label=legend, linewidth=2.0)
        ax.fill_between(range(len(y)), y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0, alpha=0.2)
    ax.grid(True)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Success')
    loc = 'best'
    leg = ax.legend(loc=loc, prop={'size': 15}, ncol=1, labels=group_legends)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    plt.savefig('test.png', bbox_inches='tight')
