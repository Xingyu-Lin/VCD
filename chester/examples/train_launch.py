import time
from chester.run_exp import run_experiment_lite, VariantGenerator

if __name__ == '__main__':

    # Here's an example for doing grid search of openai's DDPG
    #   on HalfCheetah

    # the experiment folder name
    #   the directory is defined as /LOG_DIR/data/local/exp_prefix/, where LOG_DIR is defined in config.py
    exp_prefix = 'test-ddpg'
    vg = VariantGenerator()
    vg.add('env_id', ['HalfCheetah-v2', 'Hopper-v2', 'InvertedPendulum-v2'])

    # select random seeds from 0 to 4
    vg.add('seed', [0, 1, 2, 3, 4])
    print('Number of configurations: ', len(vg.variants()))

    # set the maximum number for running experiments in parallel
    #   this number depends on the number of processors in the runner
    maximum_launching_process = 5

    # launch experiments
    sub_process_popens = []
    for vv in vg.variants():
        while len(sub_process_popens) >= maximum_launching_process:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)

        # import the launcher of experiments
        from chester.examples.train import run_task

        # use your written run_task function
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode='local',
            exp_prefix=exp_prefix,
            wait_subprocess=False
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
