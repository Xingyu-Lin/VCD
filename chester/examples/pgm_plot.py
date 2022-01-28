from chester.plotting.cplot import *
import os.path as osp
from random import shuffle

save_path = '../sac/data/plots'
dict_leg2col = {"LSP": 0, "Base": 1, "Behavior": 2}
dict_xshift = {"LSP": 4000, "Base": 0, "Behavior": 6000}


def custom_series_splitter(x):
    params = x['flat_params']
    exp_name = params['exp_name']
    dict_mapping = {'humanoid-resume-training-6000-00': 'Behavior',
                    'humanoid-resume-training-4000-00': 'LSP',
                    'humanoid-rllab/default-2019-04-14-07-04-08-421230-UTC-00': 'Base'}
    return dict_mapping[exp_name]


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


def plot_main():
    data_path = '../sac/data/mengxiong'
    plot_key = 'return-average'
    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
        color = core.color_defaults[dict_leg2col[legend]]

        y, y_lower, y_upper = get_shaded_curve(selector, plot_key, shade_type='median')
        x = np.array(range(len(y)))
        x += dict_xshift[legend]
        y = sliding_mean(y, 5)
        ax.plot(x, y, color=color, label=legend, linewidth=2.0)

        # ax.fill_between(x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0,
        #                 alpha=0.2)

        def y_fmt(x, y):
            return str(int(np.round(x))) + 'K'

        ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
        ax.grid(True)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Average-return')

        # plt.title(env_name.replace('Float', 'Push'))
        loc = 'best'
        leg = ax.legend(loc=loc, prop={'size': 20}, ncol=1, labels=group_legends)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(3.0)

        save_name = filter_save_name('plots.png')
        plt.savefig(osp.join(save_path, save_name), bbox_inches='tight')


if __name__ == '__main__':
    plot_main()
