import numpy as np
from np_gui import np_clickable_image

grey=np.array([[1,2,3],[3,4,5]])
grey=np_clickable_image.ClickableImage.constant_to_clickable(grey)

print( 'grey resized ok' if grey.resize((2,2)).shape==(2,2) else 'grey not resized ok.')

color=np.array([[[1,2,2],[1,2,2],[1,2,2]],[[1,2,2],[1,2,2],[1,2,2]]])
color=np_clickable_image.ClickableImage.constant_to_clickable(color)

print( 'color resized ok' if color.resize((2,2)).shape==(2,2,3) else 'color not resized ok.')