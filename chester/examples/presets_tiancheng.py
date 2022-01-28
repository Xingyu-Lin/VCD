# Updated By Tiancheng Jin 08/28/2018

# the preset file should be contained in the experiment folder ( which is assigned by exp_prefix )
#   for example, this file should be put in /path to project/data/local/

# Here's an example for custom_series_splitter
#   suppose we want to split five experiments with random seeds from 0 to 4 into two strategies
#       * two groups for those with odd or plural random seeds: [0,2,4] and [1,3]
#       * two groups for those with smaller or larger random seeds: [0,1,2] and [3,4]

preset_names = ['odd or plural','small or large']


def make_custom_seris_splitter(preset_name):
    legend_note = None
    custom_series_splitter = None

    if preset_name == 'odd or plural':
        # build a custom series splitter for odd or plural random seeds
        #   where the input is the data for experiment ( contains both the results and the parameters )
        def custom_series_splitter(x):
            # extract the parameters
            params = x['flat_params']
            # make up the legend
            if params['seed'] % 2 == 0:
                legend = 'odd seeds'
            else:
                legend = 'plural seeds'
            return legend

        legend_note = "Odd or Plural"

    elif preset_name == 'small or large':
        def custom_series_splitter(x):
            params = x['flat_params']
            if params['seed'] <= 2:
                legend = 'smaller seeds'
            else:
                legend = 'larger seeds'
            return legend

        legend_note = "Small or Large"
    else:
        assert NotImplementedError

    return custom_series_splitter, legend_note
