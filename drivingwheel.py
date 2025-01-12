import pygame
import time

# Initialize Pygame for Logitech Driving Wheel
def init_driving_wheel():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Detected driving wheel: {joystick.get_name()}")
        return joystick
    else:
        print("No driving wheel detected.")
        return None

# Function to read driving wheel input
def read_driving_wheel(joystick):
    if joystick is None:
        return
    pygame.event.pump()
    axis_count = joystick.get_numaxes()
    button_count = joystick.get_numbuttons()

    # for the old Driving Wheel, axis 0 is steering, axis 1 is gas, axis 2 is brake
    axes_map = {
        0: 'steering_angle',
        1: 'throttle',
        2: 'brake'
    }
    # Read axes
    for i in range(axis_count):
        axis = joystick.get_axis(i)
        # if i == 0:
        print(f"Axis {i} {axes_map[i]}: {axis:.2f}")

    # Read buttons
    for i in range(button_count):
        button = joystick.get_button(i)
        # print(f"Button {i}: {'Pressed' if button else 'Released'}")

# Main function
def main():
    # Initialize driving wheel
    joystick = init_driving_wheel()

    # Run loop to read driving wheel input
    try:
        while True:
            read_driving_wheel(joystick)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        if joystick:
            joystick.quit()
        pygame.quit()

if __name__ == "__main__":
    main()
