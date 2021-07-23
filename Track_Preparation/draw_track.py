# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 04:43:57 2020

Processes monochome track drawing to produce intermediate PNG track file images for l2race.

This script produces the intermediate PNG files that are used to produce the numpy files used in l2race track.py

track.png
track_info.npy
track_map.npy

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
plt.rcParams['figure.figsize'] = w/float(f), h/float(f)
# Name of the picture (png) we load to extract track shape
names = ['Sebring',
         'oval',
         'oval_easy',
         'track_1',
         'track_2',
         'track_3',
         'track_4',
         'track_5',
         'track_6']

# names = ['track_6']

for name in names:
    fn='./tracks_templates/'+name+'.png'
    print('Processing track {} starting from template {}'.format(name,fn))
    # Load picture
    im_original = cv.imread(fn)

    # if name == 'oval':
    #     matplotlib.use('TkAgg')
    #     plt.imshow(im_original)
    #     plt.show()
    #     matplotlib.use('Agg')
    #     pass

    # Make it grayscale
    im_original_gray = cv.cvtColor(im_original, cv.COLOR_BGR2GRAY)

    # Threshold it (convert pixels value to 255 if above some specified value, else make them 0)
    _, thresh = cv.threshold(im_original_gray, 100, 255, 0)
    # Extract contours. APPROX_SIMPLE means that we only save minimal amount of points - start and end of line segments
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Delete redundant variables
    del im_original, im_original_gray, thresh

    # Take only one contours if multiple found (the external shape)
    contour = np.squeeze(contours[1])
    # Get x,y coordinates of the pixels laying on the contour
    x = contour[:, 0]
    y = max(contour[:, 1])-contour[:, 1]  # We reverse y order to make it appropriate for matplotlib convention

    # We want to have less points to have something else than only principal directions
    x = x[::2]
    y = y[::2]

    # This lines are track specific
    # They set the starting point index and smoothly combine the end with the beginning
    # TODO document how this index is found by the track maker

    if name == 'Sebring':
        idx_start = 118  # Search for a straight part of the track to smoothly connect start and end
    elif name == 'oval':
        idx_start = 1
    elif name == 'oval_easy':
        idx_start = 1
    elif name == 'track_1':
        idx_start = 0
    elif name == 'track_2':
        idx_start = 6
    elif name == 'track_3':
        idx_start = 2
    elif name == 'track_4':
        idx_start = 170
    elif name == 'track_5':
        idx_start = 18
    elif name == 'track_6':
        idx_start = 399
    else:
        print('There is no starting point for track named {}; define the point in draw_tracks.py line 72'.format(name))

    x = np.hstack((x[idx_start:], x[:idx_start]))
    y = np.hstack((y[idx_start:], y[:idx_start]))

    # On a straight segment there is no point so we create one to make it a starting point
    x = np.insert(x, 0, (x[0]+x[-1])//2)
    y = np.insert(y, 0, (y[0]+y[-1])//2)

    # Connect first and last point by appending the copy of the start point at the end of contour
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Create a line to indicate the start
    xs = x[0]

    if name == 'oval_easy':
        dy = 80
        lw_sand = 24
        lw_asphalt = 16
    else:
        dy = 40
        lw_sand = 12
        lw_asphalt = 8

    # Plotting the track png
    fig, ax = plt.subplots()
    plt.plot(x, y, linewidth=lw_sand, color='#FFFF00', snap=True)  # Sand region (yellow)
    plt.plot(x, y, linewidth=lw_asphalt, color='#000000', snap=True)  # Asphalt (black)
    plt.plot(x, y, linewidth=1,  color='#FFFFFF', snap=True, ls='--')  # Central line (white)
    plt.plot((xs, xs), (y[0]-dy, y[0]+dy), linewidth=1,  color='#778899', snap=True)  # Start line (gray)
    # snap = True should prevent color interpolations between areas of different colors
    # It doesn't work fine, is also not crucial - we use another picture to extract information about the track
    # Maybe it makes the boundaries of colours more sharp, maybe one can remove it. Not relevant now.
    # This is the fig checking where is the starting point and direction of the contour point list
    # plt.fig(x,y,       lw=1,  color = '#000000', marker='.', snap = True)
    # plt.fig(x[0],y[0], lw=10, color = '#FFFF00', marker='o', snap = True)
    # plt.fig(x[1],y[1], lw=10, color = '#778899', marker='o', snap = True)



    # These lines display the track, but you have to first comment out matplotlib.use('Agg') at the very beginning
    # Don't left matplotlib.use('Agg') uncommented however, it may result in a different final scaling
    # ax.set_facecolor('#4169E1')  # (blue)
    # plt.show()

    # Make frame around the axes invisible
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Make ticks invisible
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Margins have to be slightly bigger than default (0.1),
    # otherwise the thick lines (like the track) are not plotted correctly
    plt.margins(.15)
    # Plot it with so little white spaces around as possible
    plt.tight_layout()
    fn='../media/tracks/'+name+'.png'
    print('saving {}'.format(fn))
    fig.savefig(fn,  transparent=True, dpi=f, pad_inches=0.0, bbox_inches='tight')

    # bbox_inches additionally cuts the white spaces but it does so after creating png - the resulting picture is smaller
    # Here we rescale it back to have dimensions (w,h)
    im = cv.imread('../media/tracks/'+name+'.png', cv.IMREAD_UNCHANGED)
    # Without cv.IMREAD_UNCHANGED you loose information about transparency
    im = cv.resize(im, (w, h))
    cv.imwrite('../media/tracks/'+name+'.png', im)

    # We now have the png with the track we will use in our game!
    # We have extracted previously the central line as plotted in the coordinates of the old picture.
    # These are not the coordinates of that line in the coordinates of the picture we just plotted
    # Even if the picture were scaled to mach the dimensions (w,h),
    # still matplotlib performs some scaling while plotting.
    # We have to extract the central line again from the final picture we use for pygame

    # Make second picture in grayscale to indicate different parts of the track:
    # 0 out of track
    # 0.2 sand area
    # 0.8 asphalt
    # 1.0 middle line

    fig, ax = plt.subplots()

    # You have to fig the invisible start line, otherwise the picture rescales
    plt.plot((xs, xs), (y[0]-dy, y[0]+dy), linewidth=1,  color='0.0', snap=True)  # Start line (gray)
    plt.plot(x, y, linewidth=lw_sand,  color='0.2', snap=True)
    plt.plot(x, y, linewidth=lw_asphalt,  color='0.8', snap=True)
    plt.plot(x, y, linewidth=0.5, color='1.0', snap=True)
    ax.set_facecolor('0.0')

    # Make frame around the axes invisible
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Make ticks invisible
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.margins(.15)
    plt.tight_layout()
    fn='./tracks_gray/'+name+'_G.png'
    print('saving grayscale frame {}'.format(fn))
    fig.savefig(fn,  transparent=False, dpi = f, pad_inches=0.0, bbox_inches='tight')

    # bbox_inches cuts the white spaces but it does so after creating png - the resulting picture is smaller
    # Here we rescale it back to have dimensions (w,h)
    im = cv.imread('./tracks_gray/'+name+'_G.png', cv.IMREAD_UNCHANGED)
    im = cv.resize(im, (w, h))
    cv.imwrite('./tracks_gray/'+name+'_G.png', im)

    # trackG.png is exactly identical to track.png except for:
    # - it is in grayscale (the grayscale is actually lost while saving to png, but we will recover it later)
    # - the middle line is continuous and as thin as possible (still not just 1 pixel)

    # We finished preparing an equivalent png, now we prepare a numpy matrix out of it - track_map and TrackInfo


    # We prepare the third picture to easily extract start position

    fig, ax = plt.subplots()

    # You have to fig the invisible start line, otherwise the picture rescales

    plt.plot(x, y, linewidth=lw_sand,  color='0.0', snap=True)
    plt.plot(x, y, linewidth=lw_asphalt,  color='0.0', snap=True)
    plt.plot(x, y, linewidth=0.5, color='0.0', snap=True)
    plt.plot((xs, xs), (y[0]-dy, y[0]+dy), linewidth=1,  color='1.0', snap=True)  # Start line (gray)
    ax.set_facecolor('0.0')

    # Make frame around the axes invisible
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Make ticks invisible
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.margins(.15)
    plt.tight_layout()
    fn='./tracks_start/'+name+'_start.png'
    print('saving starting position frame {}'.format(fn))
    fig.savefig(fn,  transparent=False, dpi = f, pad_inches=0.0, bbox_inches='tight')

    # bbox_inches cuts the white spaces but it does so after creating png - the resulting picture is smaller
    # Here we rescale it back to have dimensions (w,h)
    im = cv.imread('./tracks_start/'+name+'_start.png', cv.IMREAD_UNCHANGED)
    im = cv.resize(im, (w, h))
    cv.imwrite('./tracks_start/'+name+'_start.png', im)
    
    plt.close('all') # Close figure instances to save memory



    pass
