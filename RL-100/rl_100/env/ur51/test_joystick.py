import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
else:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick Name: {joystick.get_name()}")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.JOYBUTTONDOWN:
            print(f"Button {event.button} pressed")
        elif event.type == pygame.JOYBUTTONUP:
            print(f"Button {event.button} released")
        if event.type == pygame.JOYAXISMOTION:
            axis_value = joystick.get_axis(event.axis)
            print(f"Joystick {event.axis} moved to {axis_value:.2f}")
        if event.type == pygame.JOYHATMOTION:
            print(f"POV Hat {event.value}")

pygame.quit()
