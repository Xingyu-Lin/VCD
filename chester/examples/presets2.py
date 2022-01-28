preset_names = ['default']
x_axis = 'Epoch'
y_axis = 'Success'
FILTERED = 'filtered'

def make_custom_seris_splitter(preset_names):
    legendNote = None
    if preset_names == 'default':
        def custom_series_splitter(x):
            params = x['flat_params']
            if params['her_failed_goal_option'] is None:
                ret = 'Distance Reward'
            elif params['her_failed_goal_option'] == 'dist_behaviour':
                ret = 'Exact Match'
            else:
                ret = FILTERED
            return ret

        legendNote = None
    else:
        raise NotImplementedError
    return custom_series_splitter, legendNote


def make_custom_filter(preset_names):
    if preset_names == 'default':
        custom_seris_splitter, _ = make_custom_seris_splitter(preset_names)
        def custom_filter(x):
            legend = custom_seris_splitter(x)
            if legend == FILTERED:
                return False
            else:
                return True
            # params = x['flat_params']
            # if params['her_failed_goal_option'] != FILTERED:
            #     return True
            # else:
            #     return False
    return custom_filter

