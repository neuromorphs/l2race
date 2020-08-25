"""
Created on Tue Jul 14 04:43:57 2020

Processes track PNG files to produce the numpy files used in l2race track.py.

track.png
track_info.npy
track_map.npy

To use it, run this script.

@author: Marcin
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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

start_up_tracks = ['track_1', 'track_2', 'track_3', 'track_5']



# Functions to calculate angles between two vectors
# https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    if v1_u[0]*v2_u[1] - v1_u[1]*v2_u[0] < 0:
        angle = -angle
    return angle
    # return (angle_raw-360.0*np.rint(angle_raw/360.0))  # Shift range and reverse convention


for name in names:

    print('Now processing: ' + name)

    # Load gray version of the track picture and recover the grayscale format.
    fn='./tracks_gray/'+name+'_G.png'
    print('loading gray scale track image {}'.format(fn))
    im = cv.imread(fn, cv.IMREAD_UNCHANGED)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Make boundaries between regions sharp (matplotlib applies color interpolation between regions of different color)
    # And assign new values to the different regions:
    # 0 - water
    # 10 - sand (8 - left and 12 - right boundary)
    # 20 - asphalt (18 - left and 22 - right boundary)
    # 30 - middle line
    # 40 - checkpoints
    im[(im < 10)] = 0  # water
    im[(im < 100) & (im >= 10)] = 10  # sand
    im[(im < 255) & (im >= 100)] = 20  # asphalt
    im[(im == 255)] = 40  # middle line, t will be thirty, see below

    # Extract the middle line
    # We assume (from experience) that the middle line after boarder sharpening (above) is "never much broader" than 1 pixel
    # Thus any contour picking the points inside this line
    # will yield a very good approx. to the true 1pixel wide middle line.
    _, thresh = cv.threshold(im, 25, 255, 0)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # CHAIN_APPROX_NONE means that we save all points of the contour
    # we get 1 pixel wide continuous line

    # Extract (x,y) coordinates of the points belonging to this contour
    contour = np.squeeze(contours[0])
    x = contour[:, 0]
    y = contour[:, 1]

    # Downgrade the points lying on this newly extracting middle line to 30
    for i in range(len(x)):
        if im[y[i], x[i]] == 40:
            im[y[i], x[i]] = 30
        # We do not get other cases - this confirms that we pick the contour from the right points -
        # - inside the old middle line
        elif im[y[i], x[i]] == 2000:
            im[y[i], x[i]] = 0
        else:
            im[y[i], x[i]] = -10

    # Degrade other points from the middle line to be a normal asphalt (20)
    im[(im == 40)] = 20

    # Draw boundaries of asphalt and sand regions

    imA = np.copy(im)  # "A(sphalt)" an image copy to extract boundaries of asphalt region
    imA[imA >= 20] = 100
    imA[imA < 20] = 0

    imS = np.copy(im)  # "S(and)" an image copy to extract boundaries of sand region
    imS[imS >= 10] = 100
    imS[imS < 10] = 0


    def boundaries(im_original, im_copy, b_left, b_right, track_name='Sebring'):
        """
        Finds right and left asphalt and sand region boundaries.

        Needs hardcoding if the track has mulitple boundaries, e.g. if there are overlapping segments.

        :param im_original: TODO
        :param im_copy:
        :param b_left:
        :param b_right:
        :param track_name:
        :return:
        """
        _, thresh_b = cv.threshold(im_copy, 25, 255, 0)
        contours_b, _ = cv.findContours(thresh_b, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        contour1 = np.squeeze(contours_b[0])
        contour2 = np.squeeze(contours_b[1])

        x1 = contour1[:, 0]
        y1 = contour1[:, 1]
        x2 = contour2[:, 0]
        y2 = contour2[:, 1]

        # Check which contour is the left, and which is the right boundary of the given shape
        # We assume our tracks are clockwise. Point with biggest y will hence be on the left hand side
        # Remember that y axis is pointing down.
        if max(y1) > max(y2):
            (xl, yl, xr, yr) = (x1, y1, x2, y2)
        else:
            (xl, yl, xr, yr) = (x2, y2, x1, y1)

        # Manual correction for our sand region, if there are more contours than 2

        if len(contours_b) > 2:
            print('You have more than 2 contours in ' + track_name)
            pass
            # left: 0, 1
            # right: 3, 2
            if (track_name == 'Sebring') & (b_left == 8):
                c0 = np.squeeze(contours_b[0])
                c1 = np.squeeze(contours_b[1])
                c2 = np.squeeze(contours_b[2])
                c3 = np.squeeze(contours_b[3])
                xl = np.hstack((c0[:, 0], c1[:, 0], c2[:, 0]))
                xr = np.hstack((c3[:, 0]))
                yl = np.hstack((c0[:, 1], c1[:, 1], c2[:, 1]))
                yr = np.hstack((c3[:, 1]))

            else:
                print('You have to correct some contour!')

        # Matplotlib code to check if you combined contours correctly
        # if name == 'track_6':
        #     plt.figure()
        #     plt.plot(xl, yl, 'r.')
        #     plt.plot(xr, yr, 'b.')
        #     plt.title(track_name)
        #     plt.show()


        # Make asphalt boundaries
        for idx in range(len(xl)):
            im_original[yl[idx], xl[idx]] = b_left
        for idx in range(len(xr)):
            im_original[yr[idx], xr[idx]] = b_right

        return xl, yl, xr, yr


    boundaries(im, imA, b_left=18, b_right=22, track_name=name)
    boundaries(im, imS, b_left=8, b_right=12, track_name=name)
    del imA, imS

    # Find start line -- assume it is perfectly vertical
    # Load gray version of the start line picture and recover the grayscale format.
    im_start = cv.imread('./tracks_start/' + name + '_start.png', cv.IMREAD_UNCHANGED)
    im_start = cv.cvtColor(im_start, cv.COLOR_BGR2GRAY)

    # Make it sharp
    im_start[im_start < 200] = 0
    im_start[im_start > 0] = 100

    # find the boundary to the right
    (Y, X) = np.where(im_start > 0)

    if name in start_up_tracks:
        X_start_idx = np.where(X == min(X))  # Clockwise, start up
    else:
        X_start_idx = np.where(X == max(X))  # Clockwise, start down

    Y = Y[X_start_idx]
    X = X[X_start_idx]

    for i in range(len(X)):
        im_start[Y[i], X[i]] = 200

    im_start[im_start < 200] = 0

    idx = np.where((Y == max(Y)))
    start_max = (Y[idx], X[idx])
    idx = np.where(Y == min(Y))
    start_min = (Y[idx], X[idx])

    # Actually it is enough to save dy. The x position will be given by the first checkpoint

    # Find the point on the start line and on the middle line
    (Y, X) = np.where(((im_start > 0) & (im == 30)))
    pass

    idx_start = np.array(np.where((x == X) & (y == Y))).squeeze()

    x = np.hstack((x[idx_start:], x[:idx_start]))
    y = np.hstack((y[idx_start:], y[:idx_start]))

    # if necessry reverse order of the points with x[0] remaining x[0]
    if (y[0] < max(y)/2 and x[20] < x[0]) or (y[0] > max(y)/2 and x[20] > x[0]): # remember y-axis points down
        print('I change the direction of '+name)
        x = x[::-1]
        x = np.hstack((x[-1], x[:-1]))
        y = y[::-1]
        y = np.hstack((y[-1], y[:-1]))

    # Choose points on the middle line to make checkpoints
    x = x[::20]
    y = y[::20]

    # Make checkpoints - upgrade the chosen points on the middle line
    for i in range(len(x)):
        if im[y[i], x[i]] == 30:  # Check if these points lay on the middle lien
            im[y[i], x[i]] = 40
        else:
            print('Error while making checkpoints')


    # from matplotlib import colors
    # cmap = colors.ListedColormap(['blue', 'orange', 'yellow', 'peru', 'lightcoral', 'magenta', 'rosybrown', 'red', 'maroon'])
    # bounds=[0,7,9,11,17,19,21,27,33,40]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    #
    # plt.matshow(im[520:-100,770:-100], cmap=cmap, norm=norm)
    #
    # plt.show()

    # Calculate additional information for a user
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])

    # Calculating distance to the next checkpoint
    sLens = []
    for i in range(len(x)):
        seg = np.array((dx[i], dy[i]))
        sLen = np.linalg.norm(seg)
        sLens.append(sLen)

    del sLen
    sLens = np.array(sLens)
    # print('Maximal and minimal segment length')
    # print(max(sLens))
    # print(min(sLens))


    # Calculate cumulative distance from start
    CumSum = np.cumsum(sLens)
    # print('Total track length')
    # print(max(CumSum))


    # Calculate angle to the next checkpoint
    east = np.array((1, 0))  # The y axis points downwards
    angles = []
    for i in range(len(x)):
        segment = np.array((dx[i], dy[i]))
        angle = angle_between(east, segment)
        angles.append(angle)
    angles = np.array(angles)

    # Short segment = segment between two consecutive checkpoints
    # Find the angle between previous and following short segment
    anglesRelative = []
    for i in range(len(x)):
        if i == 0:
            segment_previous = np.array((dx[- 1], dy[- 1]))
        else:
            segment_previous = np.array((dx[i - 1], dy[i - 1]))

        segment_next = np.array((dx[i], dy[i]))
        angle = angle_between(segment_previous, segment_next)
        anglesRelative.append(angle)
    anglesRelative = np.array(anglesRelative)


    # print('Max angle single segment')
    # print(max(anglesRelative))
    # print('Min angle single segment')
    # print(min(anglesRelative))

    # Find the angle of the segment connecting
    # the points first before and first after the given point

    dx2 = []
    dy2 = []

    for i in range(len(x)):

        if i == 0:
            p1 = np.array((x[- 1], y[- 1]))
        else:
            p1 = np.array((x[i - 1], y[i - 1]))

        if i == len(x) - 1:
            p2 = np.array((x[0], y[0]))
        else:
            p2 = np.array((x[i + 1], y[i + 1]))

        (dx2s, dy2s) = tuple(p2 - p1)  # "s" for "single (value)"
        dx2.append(dx2s)
        dy2.append(dy2s)

    dx2 = np.array(dx2)
    dy2 = np.array(dy2)

    angles2 = []
    for i in range(len(x)):
        segment = np.array((dx2[i], dy2[i]))
        angle = angle_between(east, segment)
        angles2.append(angle)
    angles2 = np.array(angles2)

    TrackInfo = {'waypoint_x': x,
                 'waypoint_y': y,
                 'DistNextCheckpoint': sLens,
                 'DistTotal': CumSum,
                 'AngleNextCheckpointEast': angles,
                 'AngleNextCheckpointRelative': anglesRelative,
                 'AngleNextSegmentEast': angles2}

    # Saving all relevant data
    fn1='../media/tracks/' + name + '_map.npy'
    fn2='../media/tracks/' + name + '_info.npy'
    print('saving {} and {}'.format(fn1,fn2))
    np.save(fn1, im)
    np.save(fn2, TrackInfo)

    # Summary
    # We give to the user these track-only dependant information:
    # (1) array with: waypoints, middle line, boundary of the road and sand (left right), road, sand
    # (2) ordered list of waypoints, their x,y coordinates (apporx. every 20 pixels) with x[0] being starting/end position
    # (3) list of distances from the current waypoint to the next waypoint
    # (4) Cumulative sum - distances from start to the current waypoint
    # angle of the direction to the next waypoint
    #   - (5) related to east
    #   - (6) related to previous segment
    # Def: nearest segment
    #         is for us a segment connecting the next waypoint before and the next waypoint after the nearest waypoint
    # (7) The angle of the nearest segments with respect to east

    # As the basis for competition serves the (1) and car_state of the car
    # You are allowed to use (2)-(6) and other dynamically updated information about the car with relation to the track
    # (accessible through the Track class functions)
    # But you are welcome to calculate another metrics as well.


    # matplotlib.use('TkAgg')
    # plt.figure()
    # imgplot = plt.imshow(im)
    # plt.show()

    pass


