"""
--------------------------------
The np_clickable_image module
--------------------------------

    This module implements ClickableImage and some subclasses.

"""



from __future__ import annotations

import numpy as np
from typing import Callable, Any
from matplotlib import pyplot as plt
import skimage
import os
import sys
import inspect, warnings


from . import image_annotation
from . import colors




class _CombinableFunc:
    """A class of callables mimicking positional keywords functions.

    A class of callables mimicking functions with only positional
    keyword arguments with better control on signature and default
    values. This class does not care about the specific order of the
    positional keyword arguments.


    """

    @staticmethod
    def defaults_dictionary(f: Callable[Any, Any]) -> dict:
        """Parse the keyword(-only) arguments default values.

        Args:
            f (Callable[Any,Any]): A function.

        Returns:
            dict: A dictionary with keys given by the keyword arguments
                and keyword-only arguments of f that maps these keys to
                the corresponding default values.
        """
        data = inspect.getfullargspec(f)
        args = data.args
        defaults = data.defaults
        n = len(defaults)
        k = len(args) - n
        if k != 0:
            raise TypeError(
                "There are some positional non-keyword "
                + "arguments in your function"
            )
        args_dic = {args[k + i]: defaults[i] for i in range(n)}
        kwonlydefaults = data.kwonlydefaults
        if kwonlydefaults is None:
            kwonlydefaults = {}
        return args_dic | kwonlydefaults

    def __init__(self, func: Callable, defaults_dic: dict = None):
        """Build the callable.

            Build the callable from a  keyword(-only) function and
            optionally a dictionary. The function can be defined with
            a signature ( \*\*kwargs, \*, ...). In this case, the placeholder
            for a detailed specification of keyword
            arguments is the dictionary, see below. If the dictionary
            is not passed, the function's signature must specify every
            admissible positional keyword argument. Functions using
            also keyword-only arguments can be passed but the
            keyword-only arguments with no default value will be
            ignored. If a dictionary is passed, the function's signature
            is ignored.


        Args:
            func (Callable): A function with solely keyword and
                keyword-only arguments;
            dic (dict, optional): A dictionary specifying all admissible
                positional keyword arguments TOGETHER WITH their
                default values. Defaults to None.
        """

        if defaults_dic is None:
            defaults_dic = _CombinableFunc.defaults_dictionary(func)

        self.defaults_dic = defaults_dic
        self.function = func

    def __call__(self, **kwargs):
        for key in kwargs.keys():
            if key not in self.defaults_dic:
                raise KeyError(
                    "The argument " + key + " is not valid for this callable."
                )
        completed_args = kwargs | {
            key: self.defaults_dic[key]
            for key in self.defaults_dic
            if key not in kwargs
        }
        return self.function(**completed_args)


# # Testing _CombinableFunc
# def _f(a=2, *, u=3):
#     return a + u


# f = _CombinableFunc(_f)
# print(f())  # 5
# print(f(a=7))  # 10
# print(f(a=7, u=12))  # 19
# print(f(u=13))  # 15


# def _g(**kwargs):
#     return sum([*kwargs.values()])


# defaults = {"a": 2, "u": 3}
# g = _CombinableFunc(_g, defaults)
# print(g())  # 5
# print(g(a=7))  # 10
# print(g(a=7, u=12))  # 19
# print(g(u=13))  # 15


# A decorator to define _CombinableFuncs as functions with detailed
# signatures.
def _combinable_func(g):
    return _CombinableFunc(g)


# @_combinable_func
# def f(a=2, *, u=3):
#     return a + u


# print(f())  # 5
# print(f(a=7))  # 10
# print(f(a=7, u=12))  # 19
# print(f(u=13))  # 15
# print(f.defaults_dic)







class ClickableImage:
    """A class for clickable images, made with matplotlib and numpy."""

    def __init__(
        self,
        image: Callable[Any, np.ndarray] | np.ndarray,
        shape: tuple[int, ...],
        regions: list[np.ndarray],
        callbacks: list[Callable[dict, None]],
        vars_dic: dict,
    ):
        """Builder method

        Args:
            image (Callable[Any, np.ndarray] | np.ndarray): A function that
                returns an image or a 'constant' image. If a function is
                passed, it must have only keyword and keyword-only arguments.
                The keyword-only arguments with no default value will be
                ignored.

            shape (tuple[int, ...]): shape of the ndarray returned by
                image
            regions (list[np.ndarray]): boolean arrays A with A.shape=shape[:2]

            callbacks (list[Callable[dict, None]]): functions to be
                called when the regions are clicked
                (the ith callback belongs to the ith region)
            vars_dic (dict): some variable values stored in a dict. They
                are meant to define the behaviour of the image attributes
                of the current instance and possibly other ones, in view of
                interactions between instances (stacking).

        Raises:
            ValueError: The number of regions does not equal the
                number of callbacks.
            ValueError: The regions' shapes do not all equal shape[:2]
            ValueError: The passed regions do not have
                dtype('bool') dtype.
        """
        # Compatibility checks.
        if not len(callbacks) == len(regions):
            raise ValueError(
                "The number of regions does not equal the"
                + " number of callbacks"
            )
        if not all([region.shape == shape[:2] for region in regions]):
            raise ValueError("The regions' shapes do not all equal shape[:2]")
        if not all([region.dtype == np.dtype("bool") for region in regions]):
            raise ValueError(
                "The passed regions do not have dtype('bool') dtype."
            )
        if isinstance(image, _CombinableFunc):
            self.image = image
        elif isinstance(image, np.ndarray):
            self.image = _CombinableFunc(lambda: image, {})
        else:
            self.image = _CombinableFunc(image)
        self.shape = shape
        self.regions = regions
        self.callbacks = callbacks
        self.vars_dic = vars_dic

    def get_image(self, dic: dict | None = None) -> np.ndarray:
        if dic is None:
            dic = self.vars_dic
        keys = self.image.defaults_dic.keys()
        kwargs = {key: dic[key] for key in keys if key in dic}
        output = self.image(**kwargs)
        if output.shape != self.shape:
            raise ValueError(
                "The image attribute of your ClickableImage "
                + "did not return a ndarray with shape equal to the shape "
                + "attribute."
            )

        return output

    def click_action(self, index: int):
        callback = self.callbacks[index]
        callback(self.vars_dic)

    def use(
        self, return_vars: list[str] | None = None, **plot_options
    ) -> dict:
        """Use the clickable image to get some user input.

        For this method to work, all the keyword arguments for image
        must be contained in vars_dic.

        Args:
            return_vars (list[str] | None, optional):
                The keys of vars_dic we are interested in. Defaults to None.
                If None is received, all vars_dic will be returned after
                the interaction.

            **plot_options: the optional arguments to be passed to
                imshow. Useful to affect the display.

        Returns:
            dict: The requested dictionary of variables values.
        """
        # Display initialization.
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 5)
        ax.set_axis_off()
        ax.imshow(self.get_image(), **plot_options)

        # Event handling.
        def onclick(event):
            if ax == event.inaxes:
                y = int(event.ydata + 0.5)
                x = int(event.xdata + 0.5)

                # Variables and display update.
                for i, region in enumerate(self.regions):
                    if region[y, x]:
                        self.click_action(i)
            ax.imshow(self.get_image(), **plot_options)
            plt.pause(0.0001)

        cid = fig.canvas.mpl_connect("button_press_event", onclick)
        plt.show()

        if return_vars is None:
            return self.vars_dic
        return {key: self.vars_dic[key] for key in return_vars}

    def has_interaction(self, clickable: list[ClickableImage]):
        d1 = self.vars_dic | self.image.defaults_dic
        d2 = clickable.vars_dic | clickable.image.defaults_dic
        keys1 = set(d1.keys())
        keys2 = set(d2.keys())
        return bool(keys1.intersection(keys2))

    def transpose(self):
        def new_image_func(**kwargs):
            to_be_transposed = self.image.function(**kwargs)
            ndim = to_be_transposed.ndim
            return np.transpose(
                to_be_transposed, axes=(1, 0) + tuple(range(2, ndim))
            )

        new_image = _CombinableFunc(new_image_func, self.image.defaults_dic)
        new_regions = [np.transpose(region) for region in self.regions]
        new_shape = (self.shape[1], self.shape[0]) + self.shape[2:]

        return ClickableImage(
            new_image, new_shape, new_regions, self.callbacks, self.vars_dic
        )

    @staticmethod
    def constant_to_clickable(img: np.ndarray) -> ClickableImage:
        """Convert np.ndarray image to ClickableImage.

        Args:
            img (np.ndarray): an image

        Returns:
            ClickableImage: The corresponding "constant" ClickableImage.
        """
        return ClickableImage(img, img.shape, [], [], {})

    @staticmethod
    def hstack(
        clickables: list[ClickableImage | np.ndarray],
    ) -> ClickableImage:
        """Stack horizontally the input ClickableImages or np.ndarray images.

        Args:
            clickables (list[ClickableImage|np.ndarray]): list of
                ClickableImages or images, in the second case they are
                treated as 'constant' ClickableImages.

        Raises:
            ValueError: The clickables are not stackable as demanded, due
                to shape issues.

        Returns:
            ClickableImage: The ClickableImage obtained by stacking
                horizontally the input clickables/images, with possible
                interactions if they share keys of their
                image.defaults_dic | vars_dic
        """
        height = clickables[0].shape[0]

        for i in range(len(clickables)):
            if isinstance(clickables[i], np.ndarray):
                clickables[i] = ClickableImage.constant_to_clickable(
                    clickables[i]
                )

        def measure_depth(clickable: np.ndarray) -> int | None:
            A = clickable.get_image()
            return None if A.ndim == 2 else A.shape[2]

        depth = measure_depth(clickables[0])
        same_heights = all(
            [clickable.shape[0] == height for clickable in clickables]
        )

        same_depth = all(
            [measure_depth(clickable) == depth for clickable in clickables]
        )

        if not (same_heights and same_depth):
            raise ValueError(
                "The clickables are not stackable as demanded,"
                + " due to shape issues."
                + " Remember this could include a depth issue."
            )
        n_blocks = len(clickables)
        block_x_coords = [
            sum(clickables[i].shape[1] for i in range(j))
            for j in range(n_blocks + 1)
        ]
        width = block_x_coords[-1]
        new_shape = (height, width, depth)

        new_regions = []
        new_callbacks = []
        new_vars_dic = {}
        new_image_defaults = {}
        for i, clickable in enumerate(clickables):
            new_callbacks += clickable.callbacks
            new_vars_dic |= clickable.vars_dic
            new_image_defaults |= clickable.image.defaults_dic

            # Checking for interactions between the blocks.
            for j, other_clickable in enumerate(clickables):
                if j > i and other_clickable.has_interaction(clickable):
                    print(
                        "It is likely that you are setting up some "
                        + "interactions between the "
                        + str(i)
                        + "th and "
                        + str(j)
                        + "th passed clickables (counting from 0): "
                        + "they have common keys in their respective "
                        + "'vars_dics | image.defaults_dic'."
                    )

            # Regions processing.
            empty_left_block = np.zeros(
                (height, block_x_coords[i]), dtype="bool"
            )
            empty_right_block = np.zeros(
                (height, width - block_x_coords[i + 1]), dtype="bool"
            )
            for region in clickable.regions:
                translated_region = np.hstack(
                    [empty_left_block, region, empty_right_block]
                )
                new_regions.append(translated_region)

        # new_image callable definition

        def new_image_func(**kwargs):
            for clickable in clickables:
                blocks = [
                    clickable.get_image(kwargs) for clickable in clickables
                ]
            return np.hstack(blocks)

        # new_image definition
        new_image = _CombinableFunc(new_image_func, new_image_defaults)

        return ClickableImage(
            new_image, new_shape, new_regions, new_callbacks, new_vars_dic
        )

    @staticmethod
    def vstack(
        clickables: list[ClickableImage | np.ndarray()],
    ) -> ClickableImage:
        """Stack vertically the input ClickableImages or np.ndarray images.

        Args:
            clickables (list[ClickableImage|np.ndarray]): list of
                ClickableImages or images, in the second case they are treated
                as'constant' ClickableImages.
        Raises:
            ValueError: The clickables are not stackable as demanded,
                due to shape issues. Remember this could include a depth issue.

        Returns:
            ClickableImage: The ClickableImage obtained by stacking
                vertically the input clickables, with possible
                interactions if they share keys of their
                image.defaults_dic | vars_dic
        """
        for i in range(len(clickables)):
            if isinstance(clickables[i], np.ndarray):
                clickables[i] = ClickableImage.constant_to_clickable(
                    clickables[i]
                )
        new_clickables = [clickable.transpose() for clickable in clickables]
        return ClickableImage.hstack(new_clickables).transpose()

    def resize(
        self,
        shape: tuple[int, ...],
        anti_aliasing: bool = False,
        preserve_range: bool = False,
    ) -> ClickableImage:
        """Resize self to input shape.

        Args:
            shape (tuple[int,...]): Desired shape for the output.
            anti_aliasing (bool, optional): argument to be passed in
                the underlying skimage.resize call. Defaults to False.
            preserve_range (bool, optional): argument to be passed in
                the underlying skimage.resize call. Defaults to False.
        """

        def new_image_func(**kwargs):
            return skimage.transform.resize(
                self.image(**kwargs),
                shape,
                anti_aliasing=anti_aliasing,
                preserve_range=preserve_range,
            )

        new_image = _CombinableFunc(new_image_func, self.image.defaults_dic)
        new_shape = shape
        new_regions = [
            skimage.transform.resize(region, shape[:2])
            for region in self.regions
        ]

        return ClickableImage(
            new_image, new_shape, new_regions, self.callbacks, self.vars_dic
        )


# Tests for the ClickableImage class.
# A = np.array([[1, 1], [0, 0]], dtype="bool")
# def image(is_on=True):
#     return is_on & A
# def toggle(dic):
#     dic["is_on"] ^= True


# u = ClickableImage(image, (2, 2), [A], [toggle], {"is_on": True})

# print(u.get_image())
# u.click_action(0)
# print(u.get_image())
# u.click_action(0)
# print(u.get_image())


# v = ClickableImage.hstack([u, u])
# print(v.callbacks)
# print(v.regions)
# print(v.vars_dic)
# v.click_action(1)
# print(v.vars_dic)

# print(v.use())
# print(u.transpose().use())
# print(ClickableImage.vstack([u, u]).use(cmap="gray"))


class ImageToggles(ClickableImage):
    """A subclass of ClickableImage where regions of the image become toggles.

    No new method, a unique original method: __init__ which 'extends' the
    superclass's method."""

    def __init__(
        self,
        image: np.ndarray,
        regions: list[np.ndarray],
        *,
        daltonism=False,
        on_color="green",
        off_color="red",
        default_toggle_value=True,
        true_is_black_pixel=True,
        toggles_name="toggles",
    ):
        """Build self. Extend super().__init__, with distinct arguments.

        Args:
            image (np.ndarray): The "background" image
            regions (list[np.ndarray]): the 'toggle' regions as boolean images.
            daltonism (bool, optional): Option to set a special color mode for
                color blind people. Defaults to False.
            on_color (str, optional): The color of the regions when toggled to
                True. Defaults to "green". This string must be recognized
                by matplotlib.colors.to_rgb.
            off_color (str, optional): The color of the regions when toggled to
                False. Defaults to "red".  This string must be recognized
                by matplotlib.colors.to_rgb.
            default_toggle_value (bool, optional): The common value of all
            toggles as defined in the Clickable image vars_dic. Defaults to True.
            true_is_black_pixel (bool, optional): Decide if the front pixel
                are black, in case image is boolean. Defaults to True.
            toggles_name (str, optional): the key of the toggles values list in
                the ClickableImage's vars_dic. Defaults to "toggles".

        Raises:
            TypeError: "The input regions do not have 'bool' dtype.")

        """
        if all(region.dtype == np.dtype("bool") for region in regions):
            self.regions = regions
        else:
            raise TypeError("The input regions do not have 'bool' dtype.")

        # Defining a clickable image, we first calculate the 'image' attribute.
        if daltonism:
            on_color = "cyan"
            off_color = "magenta"
        on_color = colors.color_to_rgb(on_color)
        off_color = colors.color_to_rgb(off_color)

        my_colors = [off_color, on_color]

        def _displayed_image(**kwargs):
            toggles = kwargs[toggles_name]
            if true_is_black_pixel and image.ndim == 2:
                output = colors.to_rgb(image ^ True)
            else:
                output = colors.to_rgb(image)

            for i, region in enumerate(self.regions):
                output[region] = my_colors[toggles[i]]
            return output

        vars_dic = {toggles_name: [default_toggle_value] * len(regions)}
        displayed_image = _CombinableFunc(_displayed_image, vars_dic)

        def toggle_family(i):
            def f(dic):
                dic[toggles_name] = [
                    dic[toggles_name][j]
                    if j != i
                    else dic[toggles_name][i] ^ True
                    for j in range(len(dic[toggles_name]))
                ]

            return f

        callbacks = [toggle_family(i) for i in range(len(regions))]

        super().__init__(
            displayed_image,
            image.shape[:2] + (3,),
            regions,
            callbacks,
            vars_dic,
        )


class SliceDisplayer(ClickableImage):
    """This subclass is for clickable images that display a slice of text.

    The slice may have variable start and stop.
    """

    def __init__(
        self,
        text: str,
        shape: tuple[int, int],
        *,
        background_color="white",
        slice_varname="slice",
    ):
        """Extends __init__ of superclass, with distinct arguments.

        Args:
            text (str): The text whose slice will be displayed
            shape (tuple[int, int]): The shape of the resulting ClickableImage
            background_color (str, optional): Background color, this string
                must be recognized by matplotlib.colors.to_rgb.
                Defaults to "white".
            slice_varname (str, optional): The name of the variable that
                defines the slice to take. Defaults to "slice".

        """
        defaults_dic = {slice_varname: slice(None, 7)}

        def text_image_(**kwargs):
            return image_annotation.center_text(
                text[kwargs[slice_varname]],
                shape,
                background_color=background_color,
            )
            

        text_image = _CombinableFunc(text_image_, defaults_dic)
        regions = []
        callbacks = []
        super().__init__(
            text_image, shape + (3,), regions, callbacks, defaults_dic
        )


class Button(ClickableImage):
    """The subclass of ClickableImage made of basic rectangular buttons.

    More precisely, the unique clickable region is the whole image.
    """

    def __init__(self, image: np.ndarray, callback: Callable[dict, None]):
        """Button constructor.

        Args:
            image (np.ndarray): The underlying image of the button.
            callback (Callable): The callback triggered by clicking the button.
        """
        if image.ndim == 3 and image.shape[2] == 3:
            image[:, 0] = np.array([0, 0, 0])
            image[:, -1] = np.array([0, 0, 0])
            image[0, :] = np.array([0, 0, 0])
            image[-1, :] = np.array([0, 0, 0])
        elif image.ndim == 2:
            image[:, 0] = 0
            image[:, -1] = 0
            image[0, :] = 0
            image[-1, :] = 0
        else:
            raise ValueError(
                "The image argument is expected to be a "
                + "binary, grayscale or rgb image."
            )

        super().__init__(
            image,
            image.shape,
            [np.ones(image.shape[:2], dtype="bool")],
            [callback],
            vars_dic={},
        )
