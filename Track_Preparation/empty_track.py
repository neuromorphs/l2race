# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 04:43:57 2020

@author: Marcin
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# dpi of the track png
f = 500
# png width and height
w: int = 1024
h: int = 768
plt.rcParams['figure.figsize'] = w / float(f), h / float(f)
# Name of the picture (png) we load to extract track shape

name = 'empty'

print(name)
# Load picture
im_original = cv.imread('./tracks_templates/' + name + '.png')

# if name == 'oval':
#     matplotlib.use('TkAgg')
#     plt.imshow(im_original)
#     plt.show()
#     matplotlib.use('Agg')
#     pass

# Make it grayscale
im_original_gray = cv.cvtColor(im_original, cv.COLOR_BGR2GRAY)

# Threshold it (convert pixels value to 255 if above some specified value, else make them 0)
_, im = cv.threshold(im_original_gray, 100, 255, 0)
# Without cv.IMREAD_UNCHANGED you loose information about transparency
im = cv.resize(im, (w, h))
cv.imwrite('../media/tracks/' + name + '.png', im)

pass

# Load gray version of the track picture and recover the grayscale format.
im = np.array(im)
im = np.full_like(im, 20)
np.save('../media/tracks/' + name + '_map.npy', im)
