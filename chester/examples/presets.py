preset_names = ['default']

def make_custom_seris_splitter(preset_names):
    legendNote = None
    if preset_names == 'default':
        def custom_series_splitter(x):
            params = x['flat_params']
            # return params['her_replay_strategy']
            if params['her_replay_strategy'] == 'future':
                ret = 'RG'
            elif params['her_replay_strategy'] == 'only_fake':
                if params['her_use_reward']:
                    ret = 'FG+RR'
                else:
                    ret = 'FG+FR'
            return ret + '+' + str(params['her_clip_len']) + '+' + str(params['her_reward_choices']) + '+' + str(
                params['her_failed_goal_ratio'])

        legendNote = "Fake Goal(FG)/Real Goal(RG) + Fake Reward(FR)/Real Goal(RG) + HER_clip_len + HER_reward_choices + HER_failed_goal_ratio"
    else:
        raise NotImplementedError
    return custom_series_splitter, legendNote
