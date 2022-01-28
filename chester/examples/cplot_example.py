from chester.plotting.cplot import *
import os.path as osp


def custom_series_splitter(x):
    params = x['flat_params']
    if 'use_ae_reward' in params and params['use_ae_reward']:
        return 'Auto Encoder'
    if params['her_replay_strategy'] == 'balance_filter':
        return 'Indicator+Balance+Filter'
    if params['env_kwargs.use_true_reward']:
        return 'Oracle'
    return 'Indicator'


dict_leg2col = {"Oracle": 1, "Indicator": 0, 'Indicator+Balance+Filter': 2, "Auto Encoder": 3}
save_path = './data/plots_chester'


def plot_visual_learning():
    data_path = './data/nsh/submit_rss/submit_rss/visual_learning'

    plot_keys = ['test/success_state', 'test/goal_dist_final_state']
    plot_ylabels = ['Success', 'Final Distance to Goal']
    plot_envs = ['FetchReach', 'Reacher', 'RopeFloat']

    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    for (plot_key, plot_ylabel) in zip(plot_keys, plot_ylabels):
        for env_name in plot_envs:
            fig, ax = plt.subplots(figsize=(8, 5))
            for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
                color = core.color_defaults[dict_leg2col[legend]]
                y, y_lower, y_upper = get_shaded_curve(selector.where('env_name', env_name), plot_key,
                                                       shade_type='median')

                env_horizon = selector.where('env_name', env_name).extract()[0].params["env_kwargs"]["horizon"]
                x, _, _ = get_shaded_curve(selector.where('env_name', env_name), 'train/episode')
                x = [ele * env_horizon for ele in x]

                ax.plot(x, y, color=color, label=legend, linewidth=2.0)

                ax.fill_between(x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0,
                                alpha=0.2)

                def y_fmt(x, y):
                    return str(int(np.round(x / 1000.0))) + 'K'

                ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
                ax.grid(True)
                ax.set_xlabel('Timesteps')
                ax.set_ylabel(plot_ylabel)
                axes = plt.gca()
                if 'Rope' in env_name:
                    axes.set_xlim(left=20000)

                plt.title(env_name.replace('Float', 'Push'))
                loc = 'best'
                leg = ax.legend(loc=loc, prop={'size': 20}, ncol=1, labels=group_legends)
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(3.0)

                save_name = filter_save_name('ind_visual_' + plot_key + '_' + env_name)

                plt.savefig(osp.join(save_path, save_name), bbox_inches='tight')


def plot_state_learning():
    data_path = './data/nsh/submit_rss/submit_rss/state_learning'

    plot_keys = ['test/success_state', 'test/goal_dist_final_state']
    plot_envs = ['FetchReach', 'FetchPush', 'Reacher', 'RopeFloat']

    exps_data, plottable_keys, distinct_params = reload_data(data_path)
    group_selectors, group_legends = get_group_selectors(exps_data, custom_series_splitter)
    for plot_key in plot_keys:
        for env_name in plot_envs:
            fig, ax = plt.subplots(figsize=(8, 5))
            for idx, (selector, legend) in enumerate(zip(group_selectors, group_legends)):
                color = core.color_defaults[dict_leg2col[legend]]
                y, y_lower, y_upper = get_shaded_curve(selector.where('env_name', env_name), plot_key,
                                                       shade_type='median')
                env_horizon = selector.where('env_name', env_name).extract()[0].params["env_kwargs"]["horizon"]
                x, _, _ = get_shaded_curve(selector.where('env_name', env_name), 'train/episode')
                x = [ele * env_horizon for ele in x]
                ax.plot(x, y, color=color, label=legend, linewidth=2.0)

                ax.fill_between(x, y_lower, y_upper, interpolate=True, facecolor=color, linewidth=0.0,
                                alpha=0.2)

                def y_fmt(x, y):
                    return str(int(np.round(x / 1000.0))) + 'K'

                ax.xaxis.set_major_formatter(tick.FuncFormatter(y_fmt))
                ax.grid(True)
                ax.set_xlabel('Timesteps')
                ax.set_ylabel('Success')

                plt.title(env_name.replace('Float', 'Push'))
                loc = 'best'
                leg = ax.legend(loc=loc, prop={'size': 20}, ncol=1, labels=group_legends)
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(3.0)

                save_name = filter_save_name('ind_state_' + plot_key + '_' + env_name)

                plt.savefig(osp.join(save_path, save_name), bbox_inches='tight')


if __name__ == '__main__':
    plot_visual_learning()
    plot_state_learning()
