import os
import glfw
from multiprocessing import Process, Queue

import cv2 as cv

class VideoRecorder(object):
    '''
    Used to record videos for mujoco_py environment
    '''
    def __init__(self, env, saved_path='./data/videos/', saved_name='temp'):
        # Get rid of the gym wrappers
        if hasattr(env, 'env'):
            env = env.env
        self.viewer = env._get_viewer()
        self.saved_path = saved_path
        self.saved_name = saved_name
        # self._set_filepath('/tmp/temp%07d.mp4')
        saved_name += '.mp4'
        self._set_filepath(os.path.join(saved_path, saved_name))

    def _set_filepath(self, video_name):
        self.viewer._video_path = video_name

    def start(self):
        self.viewer._record_video = True
        if self.viewer._record_video:
            fps = (1 / self.viewer._time_per_render)
            self.viewer._video_process = Process(target=save_video,
                                                 args=(self.viewer._video_queue,
                                                       self.viewer._video_path, fps))
            self.viewer._video_process.start()

    def end(self):
        self.viewer.key_callback(None, glfw.KEY_V, None, glfw.RELEASE, None)

# class VideoRecorderDM(object):
#     '''
#     Used to record videos for dm_control based environments
#     '''
#     def __init__(self, env, saved_path='./data/videos/', saved_name='temp'):
#         self.saved_path = saved_path
#         self.saved_name = saved_name
#
#     def