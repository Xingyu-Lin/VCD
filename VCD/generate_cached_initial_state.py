import copy
from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict
from VCD.main import get_default_args


def create_env(args):
    assert args.env_name == 'ClothFlatten'

    env_args = copy.deepcopy(env_arg_dict[args.env_name])  # Default args
    env_args['cached_states_path'] = args.cached_states_path
    env_args['num_variations'] = args.num_variations
    env_args['use_cached_states'] = False
    env_args['save_cached_states'] = True

    env_args['render'] = False
    env_args['headless'] = True
    env_args['render_mode'] = 'cloth' if args.gen_data else 'particle'
    env_args['camera_name'] = 'default_camera'
    env_args['camera_width'] = 360
    env_args['camera_height'] = 360

    env_args['num_picker'] = 2  # The extra picker is hidden and does not really matter
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    env_args['action_repeat'] = 1

    if args.partial_observable and args.gen_data:
        env_args['observation_mode'] = 'cam_rgb'

    return SOFTGYM_ENVS[args.env_name](**env_args)


if __name__ == '__main__':
    args = get_default_args()
    env = create_env(args)
