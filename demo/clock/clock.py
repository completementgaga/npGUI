#!/usr/bin/env python3

import sys
import os
import numpy as np
import skimage
import time

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, "../.."))
from np_gui import np_clickable_image, image_annotation, colors

dir_path = os.path.join(this_dir, "clock_pictures")

forward_button = colors.to_rgb(
    skimage.io.imread(os.path.join(dir_path, "forward_button.png"))
)
big_arrow = colors.to_rgb(
    skimage.io.imread(os.path.join(dir_path, "big_arrow.png"))
)
small_arrow = colors.to_rgb(
    skimage.io.imread(os.path.join(dir_path, "small_arrow.png"))
)

# image compression
forward_button = forward_button[::2, ::2]
big_arrow = big_arrow[::2, ::2]
small_arrow = small_arrow[::2, ::2]

backward_button = forward_button[:, ::-1]

background_color = np.array(
    [
        np.amax(forward_button[:, :, 0]),
        np.amax(forward_button[:, :, 1]),
        np.amax(forward_button[:, :, 2]),
    ]
)


# The callable that returns the clock image.
def clock_image(seconds=0):
    big_arrow_black_pixels = big_arrow != background_color
    small_arrow_black_pixels = small_arrow != background_color
    current_time_black_pixels = skimage.transform.rotate(
        big_arrow_black_pixels, -(seconds / 10)
    ) | skimage.transform.rotate(small_arrow_black_pixels, -seconds // 120)
    output = np.array(
        [[background_color] * big_arrow.shape[1]] * big_arrow.shape[0]
    )
    output *= 1 - current_time_black_pixels.astype("uint8")
    return output


def backward_button_callback(dic):
    dic["seconds"] -= 60


def forward_button_callback(dic):
    dic["seconds"] += 60


clock_block = np_clickable_image.ClickableImage(
    clock_image,
    (278, 280, 3),
    [],
    [],
    {"seconds": 0},
)

forward_button_block = np_clickable_image.ClickableImage(
    forward_button,
    (101, 121, 3),
    [np.ones((101, 121), dtype="bool")],
    [forward_button_callback],
    {},
)

backward_button_block = np_clickable_image.ClickableImage(
    backward_button,
    (101, 121, 3),
    [np.ones((101, 121), dtype="bool")],
    [backward_button_callback],
    {},
)

filling_block = np.array([[background_color] * 38] * 101)
lower_block = np_clickable_image.ClickableImage.hstack(
    [backward_button_block, filling_block, forward_button_block]
)

explanation_block = image_annotation.center_text(
    "Click the arrows \nto set the clock!",
    lower_block.shape[:2],
    color="maroon",
    background_color=background_color,
)

# Stacking all the blocks to get the GUI.
clock_game = np_clickable_image.ClickableImage.vstack(
    [clock_block, lower_block, explanation_block]
)


# Using the GUI.
secs = clock_game.use()["seconds"]
print(
    "You set the time to",
    (secs // 3600) % 12,
    " hour(s) ",
    (secs // 60) % 60,
    " minute(s).",
)


time.sleep(5)
for i in range(1, 0, -1):
    print("Closing in " + str(i * 5) + " seconds.")
    time.sleep(5)

print("Ciao!")
time.sleep(2)
