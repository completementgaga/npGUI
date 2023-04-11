"""
--------------------------------
The image_annotation module
--------------------------------



    This module deals with displaying text on a Numpy image.

"""

import skimage
import numpy as np
import warnings
import os
import csv

from . import colors


absolute_path = os.path.dirname(__file__)


png_alphabet_folder = os.path.join(
    absolute_path, "alphabet", "liberation_serif"
)
path_to_csv = os.path.join(png_alphabet_folder, "metadata.csv")
characters_as_images = {}
with open(path_to_csv, "r", newline="") as csvfile:
    reader = csv.reader(csvfile, dialect="excel")
    for row in reader:
        png_path = os.path.join(png_alphabet_folder, row[0])
        png_char = skimage.io.imread(png_path, as_gray=True).astype("bool")
        png_char ^= True
        character = row[1]
        characters_as_images[character] = png_char

characters_as_images[" "] = np.ones(
    characters_as_images["e"].shape, dtype="bool"
)


# This will be convenient to have all characters of the exact same
#  height and to have a small white margin on the left of
#  all characters.

max_height = max(val.shape[0] for val in characters_as_images.values())
max_width = max(val.shape[1] for val in characters_as_images.values())

for c in characters_as_images:
    shape_ = characters_as_images[c].shape
    diff = max_height - shape_[0]
    top = diff // 2
    bottom = diff - top
    characters_as_images[c] = np.vstack(
        [
            np.ones((top, shape_[1]), dtype="bool"),
            characters_as_images[c],
            np.ones((bottom, shape_[1]), dtype="bool"),
        ]
    )
    characters_as_images[c] = np.hstack(
        [
            characters_as_images[c],
            np.ones((max_height, max_width // 10 + 1), dtype="bool"),
        ]
    )


def string_to_image(s: str, relative_line_spacing: float = 0.1) -> np.ndarray:
    """Produce image of the given string.

    Args:
        s (str): The string that must appear on the image.
        relative_line_spacing (float, optional): The line spacing, relative
            to the height of ONE line. Defaults to 0.1.

    Returns:
        np.ndarray: boolean image displaying the input string. The text
        corresponding to null pixels.
    """
    if not s:
        print("The input string is empty, returning None.")
        return None
    else:
        lines = s.split("\n")
        image_lines = []
        max_width = 0

        for line in lines:
            image_lines.append(
                np.hstack([characters_as_images[c] for c in line])
            )
            max_width = max(max_width, image_lines[-1].shape[1])

        for i in range(len(image_lines)):
            line = image_lines[i]
            height, width = line.shape
            line_spacing = int(height * relative_line_spacing)
            image_lines[i] = np.hstack(
                [line, np.ones((height, max_width - width), dtype="bool")]
            )

            if i != len(image_lines) - 1:
                line = image_lines[i]
                image_lines[i] = np.vstack(
                    [line, np.ones((line_spacing, max_width), dtype="bool")]
                )

        return np.vstack(image_lines)



# currently unused function
def _annotated_image(
    image,
    text,
    x,
    y,
    size=None,
    relative_size=0.1,
    color=(0, 0, 0),
    background_color=(255, 255, 255),
    relative_padding=0.05,
):
    text_image = (1 - string_to_image(text)).astype("bool")

    # We parse colors
    color = colors.color_to_rgb(color)
    background_color = colors.color_to_rgb(background_color)


    # We deal with size issues
    position_issue = (
        "The required position for your text ("
        + text
        + ")is outside the image or too close to its border, "
        + "we push the text inside the image and scale it to fit in."
    )
    size_issue = (
        "Your text ("
        + text
        + ") is too high or too wide to fit in the image at the required "
        + "position, we adapt its size to fit in."
    )

    if size is not None:
        relative_size = size / image.shape[0]

    has_position_issue = False
    if y > image.shape[0] - 7:
        y = max(0, image.shape[0] - 7)
        has_position_issue = True
    if y < 7:
        y = min(7, image.shape[0])
        has_position_issue = True

    if x > image.shape[1] - 7:
        x = max(0, image.shape[1] - 7)
        has_position_issue = True
        if x < 7:
            x = min(7, image.shape)
            has_position_issue = True
    if has_position_issue:
        warnings.warn(position_issue)
    has_size_issue = False
    if y + image.shape[0] * relative_size > image.shape[0]:
        relative_size = 0.9 * (image.shape[0] - y) / image.shape[0]
        has_size_issue = True

    text_height = image.shape[0] * relative_size
    scaling_factor = text_height / text_image.shape[0]
    text_image = skimage.transform.rescale(text_image, scaling_factor)

    if text_image.shape[1] + x > image.shape[1]:
        scaling_factor = 0.9 * (image.shape[1] - x) / text_image.shape[1]
        text_image = skimage.transform.rescale(text_image, scaling_factor)
        has_size_issue = True

    if has_size_issue:
        warnings.warn(size_issue)

    height, width = text_image.shape

    padding = int(height * relative_padding)

    if len(image.shape) != 3:
        new_image = skimage.color.gray2rgb(image)
    else:
        new_image = image.copy()

    text_mask = np.zeros(new_image.shape[:2], dtype="bool")
    text_mask[y : y + height, x : x + width] = text_image

    if background_color is not None:
        new_image[
            y - padding : y + height + padding,
            x - padding : x + width + padding,
        ] = np.array(background_color)

    new_image[text_mask] = np.array(color)

    return new_image


def center_text(
    text: str, shape: tuple[int, int], color="black", background_color="white"
):
    """Center given text over a monochrome background to create a np.ndarray.

    Args:
        text (str): The text to print
        shape (tuple[int,int]): The shape of the background
        color: The color of the background, of a type
            acceptable by colors.color_to_rgb. Defaults to black.
        background_color: The color of the background, of a type
            acceptable by colors.color_to_rgb. Defaults to white.

    Returns:
        np.ndarray: The resulting image as an rgb np.ndarray of dtype 'uint8'.
            The text fills 90% of the width or of the height of the input shape,
            depending on its own proportions.
    """
    if text == "":
        text = " "

    # We parse colors
    color = colors.color_to_rgb(color)
    background_color = colors.color_to_rgb(background_color)

    img = np.array([[background_color] * shape[1]] * shape[0])
    text_zone = string_to_image(text) ^ True
    supp = np.where(text_zone == 0)
    slice_x = slice(np.amin(supp[1]), np.amax(supp[1]))
    slice_y = slice(np.amin(supp[0]), np.amax(supp[0]))
    text_zone = text_zone[slice_y, slice_x]

    acceptable_width = (9 * shape[1]) / 10
    acceptable_height = (9 * shape[0]) / 10

    scaling_factor = min(
        acceptable_height / text_zone.shape[0],
        acceptable_width / text_zone.shape[1],
    )

    text_zone = skimage.transform.rescale(text_zone, scaling_factor)
    h = text_zone.shape[0]
    w = text_zone.shape[1]
    x = (shape[1] - w) // 2
    y = (shape[0] - h) // 2
    text_mask = np.zeros(shape[:2], dtype="bool")
    text_mask[y : y + h, x : x + w] = text_zone

    img[text_mask] = color

    return img


# For testing purposes
# from matplotlib import pyplot as plt


# def plot_page(page, cmap="gray_r"):
#     plt.figure(figsize=(20, 20))
#     plt.imshow(page, cmap=cmap)
#     plt.show()


# img = center_text(
#     "ah, ah ah,\n Oh où vas Tu? Não quer ficar?",
#     (400, 700),
#     color="purple",
#     background_color="orange",
# )
# plot_page(img)
# img = center_text(
#     "ah, ah ah,\n Oh où vas Tu? Não quer ficar?",
#     (400, 700),
#     color="purple",
#     background_color=[20,90,165],
# )
# plot_page(img)
