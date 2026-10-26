"""Exercise the real capture loop with hardware boundaries replaced."""
import ast
import asyncio
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

SOURCE = Path(__file__).parents[1] / 'kinect-service/main.py'


def harness(enabled):
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in {'State', 'capture_loop', 'get_status'}]
    for n in nodes:
        n.decorator_list = []
    pose = Mock()
    pose.detect_for_video.return_value = SimpleNamespace(pose_landmarks=[])
    factory = Mock(return_value=pose) if enabled else Mock(side_effect=AssertionError('Pose engine started in camera-only mode'))
    env = dict(POSE_ENABLED=enabled, threading=threading, Optional=object,
               time=time, MODEL_PATH='unused', JPEG_Q=[],
               LandmarkSmoother=Mock(), render_depth=lambda depth: depth,
               get_video=lambda fmt: ('rgb', 0), auto_video_mode=Mock(),
               cv2=SimpleNamespace(COLOR_RGB2BGR=1, cvtColor=lambda rgb, _: rgb,
                                  imencode=lambda *a: (True, SimpleNamespace(tobytes=lambda: b'jpeg'))),
               freenect=SimpleNamespace(VIDEO_RGB=1, VIDEO_IR_8BIT=2, sync_get_depth=lambda: ('depth', 0)),
               vision=SimpleNamespace(PoseLandmarker=SimpleNamespace(create_from_options=factory),
                                      PoseLandmarkerOptions=lambda **kw: kw, RunningMode=SimpleNamespace(VIDEO=1)),
               mp_python=SimpleNamespace(BaseOptions=lambda **kw: kw),
               mp=SimpleNamespace(Image=lambda **kw: kw, ImageFormat=SimpleNamespace(SRGB=1)))
    # State annotations require subscriptable Optional.
    from typing import Optional
    env['Optional'] = Optional
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), env)
    state = env['State']()
    state.video_mode_pref = 'rgb'
    update = state.update
    def one_frame(*args, **kw):
        update(*args, **kw)
        raise KeyboardInterrupt
    state.update = one_frame
    env['state'] = state
    return env, state, factory, pose


class CameraOnlyTests(unittest.TestCase):
    def test_disabled_pose_keeps_camera_frame_without_starting_model(self):
        env, state, factory, _ = harness(False)
        with self.assertRaises(KeyboardInterrupt):
            env['capture_loop']()
        factory.assert_not_called()
        self.assertEqual(state.raw_jpeg, b'jpeg')
        self.assertEqual(state.rgb_jpeg, b'jpeg')
        self.assertIsNone(state.presence)
        self.assertEqual(state.frame_id, 1)

    def test_enabled_pose_still_performs_inference(self):
        env, state, factory, pose = harness(True)
        with self.assertRaises(KeyboardInterrupt):
            env['capture_loop']()
        factory.assert_called_once()
        pose.detect_for_video.assert_called_once()
        self.assertIs(state.presence, False)

    def test_status_distinguishes_disabled_pose_from_absence(self):
        env, _, _, _ = harness(False)
        status = asyncio.run(env['get_status']())
        self.assertIs(status.get('pose_enabled'), False)
        self.assertIsNone(status['presence'])


if __name__ == '__main__':
    unittest.main()
