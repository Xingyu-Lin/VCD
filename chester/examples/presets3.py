set1 = 'identity_ratio+her_clip(dist_behavior and HER)'
preset_names = [set1]
FILTERED = 'filtered'


def make_custom_seris_splitter(preset_names):
    legendNote = None
    if preset_names == set1:
        def custom_series_splitter(x):
            params = x['flat_params']
            if params['her_failed_goal_option'] in ['dist_G', 'dist_policy']:
                return FILTERED
            if params['her_identity_ratio'] is not None:
                return 'IR: ' + str(params['her_identity_ratio'])
            if params['her_clip_len'] is not None:
                return 'CL: ' + str(params['her_clip_len'])
            return 'HER'

        legendNote = 'IR: identity ratio; CL: clip length'
    else:
        raise NotImplementedError
    return custom_series_splitter, legendNote


def make_custom_filter(preset_names):
    if preset_names == set1:
        custom_seris_splitter, _ = make_custom_seris_splitter(preset_names)

        def custom_filter(x):
            legend = custom_seris_splitter(x)
            if legend == FILTERED:
                return False
            else:
                return True
    return custom_filter

