# # from pynput import keyboard
# import pygame
# import time
# import threading
# import numpy as np

# class KeyboardSupervisor(object):
#     def __init__(self, env):
#         self.listener = keyboard.Listener(
#             lambda key: KeyboardSupervisor._on_press(key, env)
#         )

#     def start(self):
#         self.listener.start()

#     def stop(self):
#         self.listener.stop()

#     @staticmethod
#     def _on_press(key, env):
#         try:
#             key = key.char
#         except:
#             key = key
#         finally:
#             if key is None:
#                 pass
#             elif key == keyboard.Key.esc:
#                 print("ESC Pressed: Supervisor stop environment.")
#                 env.emergency_stop()
#             elif '1' <= key <= '9':
#                 print(f"{key} Pressed: Moving.")
#                 env.teleop_keyboard(key)
#             else:
#                 pass


# class JoystickSupervisor(object):
#     def __init__(self, env, dt):
#         self.listener = threading.Thread(target=self._on_press, args=(env, dt), daemon=True)

#     def start(self):
#         self.listener.start()

#     def stop(self):
#         self.listener.join()

#     def _on_press(self, env, dt):
#         pygame.init()
#         pygame.joystick.init()
#         assert pygame.joystick.get_count() >= 1, "No joystick connected!"
#         joystick = pygame.joystick.Joystick(0)
#         joystick.init()

#         while True:
#             for event in pygame.event.get():
#                 if event.type == pygame.JOYBUTTONDOWN:
#                     if event.button == 8:  # HOME
#                         env.reset()
#                     elif event.button == 0:  # A
#                         env.teleop_start()
#                     elif event.button == 1:  # B
#                         env.teleop_stop()
#                     elif event.button == 2:  # X
#                         env.deploy()
#             axis_x = joystick.get_axis(3)
#             axis_y = joystick.get_axis(4)
#             if abs(axis_x) >= 1e-3 or abs(axis_y) >= 1e-3:
#                 delta_x = np.clip((axis_x + axis_y) / np.sqrt(2), -1, 1)
#                 delta_y = np.clip((axis_x - axis_y) / np.sqrt(2), -1, 1)
#                 env.teleop((delta_x, delta_y))
