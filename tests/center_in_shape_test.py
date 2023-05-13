import sys
import os

import numpy as np

import np_gui


basic_image=np.zeros((5,6,3),dtype='uint8')
shape=(50,30)

my_clickable=np_gui.np_clickable_image.ClickableImage.constant_to_clickable(basic_image)
result=my_clickable.center_in_shape(shape,frame_color='green')

print( 'shape test ok' if result.shape[:2]==shape else 'shape test not ok')
