""" 
-----------------------
The colors module
-----------------------

Provide a few color conversion facilities.

The reference color format here is the one of a numpy array
with shape (3,) and dtype np.uint8. Beware that this is not the same 
convention as in matplotlib.colors. In this package, interacting with 
this package, it is recommended to always prefer our colors.color_to_rgb
to matplotlib.colors.to_rgb.


Otherwise, you could run in incompatibily issues.


"""


from typeguard import typechecked
import numpy as np
from matplotlib import colors



# Below, the colors are supposed to be given as rgb triples.
# or that matplotlib.colors can interprete them as colors,
# that is, the colors.to_rgb function accepts them as input.


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
        color = colors.to_rgb(color)
    if isinstance(color, np.ndarray):
        color=list(map(int,list(color)))
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
