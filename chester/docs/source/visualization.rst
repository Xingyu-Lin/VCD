.. _visualization:


***************
Visualization
***************

.. _Batch-Plotting:

Batch Plotting
==============

If the data are logged with Chester, they can also be easily plotted in batch.
After the data are logged, for each experiment, the hyper-parameters are stored in ``variants.json`` and different
key values are stored in ``progress.csv``. ``chester/plotting/cplots.py`` offers the functions that can be used to
organize different experiments based on their key values:

 - ``reload_data()``: Iterate through the data folder and organize each experiment into a list, with their progress data, hyper-parameters and also analyze all the curves and give the distinct hyper-parameters.
 - ``get_group_selectors()``: You should write a ``custom_series_splitter()``, which provides a legend for each experiment based on its hyper-parameters. This function will then group all the experiments by their legends.
 - ``get_shaded_curve()``: Create the needed y-values for plots with shades (representing variance or median) for a certain key value.

A data structure from rllab visualization kit can be useful: ``Selector``. It can be constructed from the loaded
experiments data structure::

    from rllab.viskit import core
    exps_data, plottable_keys, distinct_params = reload_data(path_to_data_folder)
    selector = Selector(exps_data)

After that, it can be used to extract progress infomation for a certain key value::

    progresses = [exp.progress.get(key)) for exp in selector.extract()]

or be filtered based on certain hyper-parameters::

    selector = selector.where('env_name', env_name)

Some examples can be found in both ``chester/cplots.py`` and ``chester/examples/cplot_example.py``

.. _Interactive-Frontend:

Interactive Frontend
====================

Currently the interactive visualization feature is still coupled with the rllab.
It can be accessed by doing::

    python rllab/viskit/frontend.py <data_folder>

.. _Preset:

Preset
------
You may want to use a complex legend post-processor or splitter.
The preset feature can be used to save such a setting. First write a ``presets.py``. Then, put it in the root of the
data folder that you want to visualize. Now when you use the frontend visualization, there will be a preset button that
you can choose. Some exmples of ``presets.py`` can be found at ``chester/examples``
