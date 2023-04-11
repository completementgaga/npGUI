#!/usr/bin/env python3

import sys
import os
import numpy as np
import skimage
import time

this_dir = os.path.dirname(os.path.abspath(__file__))

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, "../.."))
from np_gui import np_clickable_image, colors

dir_path = os.path.join(this_dir, "pictures")

pictures = []
names = os.listdir(dir_path)
for path in names:
    pictures.append(
        colors.to_rgb(skimage.io.imread(os.path.join(dir_path, path)))
    )


def upper_left_image(upper_left_count=0):
    n = len(pictures)
    return pictures[upper_left_count % n][:256, :256]


def upper_right_image(upper_right_count=0):
    n = len(pictures)
    return pictures[upper_right_count % n][:256, 256:]


def lower_right_image(lower_right_count=0):
    n = len(pictures)
    return pictures[lower_right_count % n][256:, 256:]


def lower_left_image(lower_left_count=0):
    n = len(pictures)
    return pictures[lower_left_count % n][256:, :256]


def upper_left_callback(dic):
    dic["upper_left_count"] += 1


def lower_left_callback(dic):
    dic["lower_left_count"] += 1


def upper_right_callback(dic):
    dic["upper_right_count"] += 1


def lower_right_callback(dic):
    dic["lower_right_count"] += 1


ul_block = np_clickable_image.ClickableImage(
    upper_left_image,
    (256, 256, 3),
    [np.ones((256, 256), dtype="bool")],
    [upper_left_callback],
    {"upper_left_count": 0},
)

ur_block = np_clickable_image.ClickableImage(
    upper_right_image,
    (256, 256, 3),
    [np.ones((256, 256), dtype="bool")],
    [upper_right_callback],
    {"upper_right_count": 1},
)

lr_block = np_clickable_image.ClickableImage(
    lower_right_image,
    (256, 256, 3),
    [np.ones((256, 256), dtype="bool")],
    [lower_right_callback],
    {"lower_right_count": 3},
)

ll_block = np_clickable_image.ClickableImage(
    lower_left_image,
    (256, 256, 3),
    [np.ones((256, 256), dtype="bool")],
    [lower_left_callback],
    {"lower_left_count": 2},
)

upper_block = np_clickable_image.ClickableImage.hstack([ul_block, ur_block])
lower_block = np_clickable_image.ClickableImage.hstack([ll_block, lr_block])
puzzle = np_clickable_image.ClickableImage.vstack([upper_block, lower_block])

dic = puzzle.use()

val = dic["upper_left_count"]

if all((dic[key] % 4) == (val % 4) for key in dic):
    print("You completed the puzzle and obtained the image " + names[val])
else:
    print("Why didn't you solve this wonderful puzzle !?")

time.sleep(5)
for i in range(1, 0, -1):
    print("Closing in " + str(i * 5) + " seconds.")
    time.sleep(5)

print("Ciao!")
time.sleep(2)
