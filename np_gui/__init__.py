""" 
+++++++++++++++++++++
The np_gui package
+++++++++++++++++++++

This package implements Graphical User Interfaces from a NumPy user
perspective: every image corresponds to an ndarray.

The core class is \
ClickableImage whose instances' attributes are: 


* a dictionary of parameter values,
* a callable that returns an image of fixed shape,
* regions in that shape defined as boolean images of the same shape 
* and their corresponding callbacks.

When calling the use method of an instance,
the dictionary is passed to the callable, that uses (some of) its values
as arguments and outputs an image which is displayed in a window.

Upon clicking on the distinguished regions, their callbacks are called
and alter the parameter values. The image is then refreshed from this 
input.

When closing the window, the use method returns the dictionary with
its altered values; whence the interaction with the user.

These objects can be stacked vertically and horizontally to compose more
complex ones. A certain number of subclasses are proposed as basic 
building blocks. The stacked blocks may have parameter keywords in common,
which allows for interactions between them.

The code is organized in three submodules as follows.

* **np_clickable_images** contains the definition of the ClickableImage \
class and some subclasses,
* **colors** contains some color management facilities and
* **image_annotation** regards putting text in an image.


"""