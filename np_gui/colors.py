""" 
-----------------------
The colors module
-----------------------

Provide a few color conversion, color picking facilities.

The reference color format here is the one of a numpy array
with shape (3,) and dtype np.uint8. Beware that this is not the same 
convention as in matplotlib.colors. In this package, interacting with 
this package, it is recommended to always prefer our colors.color_to_rgb
to matplotlib.colors.to_rgb.


Otherwise, you could run in incompatibily issues.


"""


from typeguard import typechecked
import numpy as np
from matplotlib import colors as mcolors


# Below, the colors are supposed to be given as rgb triples.
# or that matplotlib.colors can interpret them as colors,
# that is, the matplotlib.colors.to_rgb function accepts them as input.


@typechecked
def color_to_rgb(
    color: tuple[int, int, int]
    | list[int]
    | tuple[float, float, float]
    | list[float]
    | str
    | np.ndarray
) -> np.ndarray:
    """Convert matplotlib color to 8-bits rgb color.

    Args:
        color (tuple[int, int, int] | list[int] | \
            tuple[float, float, float] |\
            list[float] | str | np.ndarray):an 8-bits rgb color or its float \
            rescaled version, with values in [0,1] or a string recognized \
            by matplotlib.to_rgb.

    Raises:
        ValueError: "Expecting a 3-channels color (rgb) or a string."
        ValueError: "To specify your color by a triple of integers,
            they should belong to the closed interval [0,255]."

    Returns:
        np.ndarray: an np.ndarray of shape (3,) and dtype 'uint8', representing
            the rgb color.
    """
    if isinstance(color, str):
        color = mcolors.to_rgb(color)
    if isinstance(color, np.ndarray):
        color = list(map(int, list(color)))
    if len(color) != 3:
        raise ValueError("Expecting a 3-channels color (rgb) or a string.")
    if isinstance(color[0], float):
        return np.round(255 * np.array(color)).astype("uint8")

    if isinstance(color[0], int):
        if -1 < color[0] < 256 and -1 < color[1] < 256 and -1 < color[2] < 256:
            return np.array(color)
        else:
            raise ValueError(
                "To specify your color by a triple of integers, "
                + "they should belong to the closed interval [0,255]."
            )


def binary2grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a binary image to rgb image.

    Args:
        image (np.ndarray): A binary image as a 2d np.ndarray

    Returns:
        np.ndarray: the corresponding grayscale image.
    """
    return 255 * image.astype("uint8")


def grayscale2rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to rgb image.

    Args:
        image (np.ndarray): A grayscale image as a 2d np.ndarray with
            dtype 'uint8'.

    Returns:
        np.ndarray: the corresponding rgb image.
    """
    return np.stack([image] * 3, axis=2)


def binary2rgb(image: np.ndarray) -> np.ndarray:
    """Convert a binary image to rgb image.

    Args:
        image (np.ndarray): A binary image as a 2d np.ndarray

    Returns:
        np.ndarray: the corresponding rgb image
    """
    return grayscale2rgb(binary2grayscale(image))


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image to the rgb format.

    Args:
        image (np.ndarray): The image to be converted.

    Raises:
        ValueError: The expected dtype of the ndarray is either 'bool'
            or 'uint8'. The expected shape is either 2d or 3d with
            shape[2]==3.

    Returns:
        np.ndarray: The image converted to the rgb format.
    """
    if image.dtype == np.dtype(bool):
        return binary2rgb(image)
    if image.dtype == np.dtype("uint8") and len(image.shape) == 2:
        return grayscale2rgb(image)
    if (
        image.dtype == np.dtype("uint8")
        and len(image.shape) == 3
        and image.shape[2] == 3
    ):
        return image
    else:
        raise ValueError(
            "The expected dtype of the ndarray is either 'bool' or 'uint8'."
            + "The expected shape is either 2d or 3d with shape[2]==3."
        )


# The function below could certainly be optimized to accelerate the loop
def main_color(
    image: np.ndarray, region: np.ndarray | None = None
) -> np.ndarray:
    """Return the most frequent color of image within region.

    Args:
        image (np.ndarray): The 2d image to be studied as 2d or 3d np.ndarray.
        region (np.ndarray | None , optional): The region in which the frequency
            should be calculated, as a boolean 2d image. Defaults to None.
            if None is passed, the full image is considered.

    Raises:
        ValueError: "region has no front pixel!"

    Returns:
        np.ndarray: The most frequent color in image within region, as
        an np.ndarray (most likely a 1,3 or 4 entries 1d array)
    """
    if image.ndim == 2:
        image = image.reshape(image.shape + (1,))
    if region is None:
        region = np.ones(image.shape[:2], dtype="bool")
    if not np.any(region):
        raise ValueError("region has no front pixel!")
    found_colors = []
    color_counts = []
    for p in zip(*np.where(region)):
        current_color = list(image[p])
        if current_color in found_colors:
            current_index = found_colors.index(current_color)
            color_counts[current_index] += 1
        else:
            found_colors.append(current_color)
            color_counts.append(1)
    max_count = max(color_counts)
    main_index = color_counts.index(max_count)
    main_color = found_colors[main_index]

    if len(main_color) == 1:
        return main_color[0]
    else:
        return np.array(main_color)


def mono_block(shape: tuple[int, int], color) -> np.ndarray:
    """Return rgb image of the given shape and color.

    Args:
        shape (tuple[int,int]): The 2d shape of the sought block

        color: A color, as accepable by color_to_rgb


    Returns:
        np.ndarray: The sought monochrome image as an rgb np.ndarray.
    """
    color = color_to_rgb(color)
    output = np.zeros(shape + (3,), dtype="uint8")
    output[:, :] = color

    return output
