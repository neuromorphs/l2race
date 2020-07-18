
import logging

import pygame
# import svglib
# from svglib.svglib import svg2rlg
from src.globals import SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS
import cmath
import numpy as np
from svgpathtools import svg2paths

logger = logging.getLogger(__name__)


class TrackX:
    """
    race track
    """

    def __init__(self, file='media/Artboard 1.svg'):
        self.vertices=None  # list of track center vertices, as [ [x1,y1], [x2,y2], ....] in game screen units
        self.centerOfLinePoints=None
        self.widths=100 # list of track width in screen pixels for each segment in vertices list
        self.kdVertices=None
        self.paths=None # original paths in svg file
        self.attributes=None # original attributes
        self.boundingBox=None # overall bounding box of track in screen pixels TODO make it so, scaling wrong now
        self.aPointList=None
        self.bPointList=None
        self.load(file)

    def draw(self, surface: pygame.surface):
        surface.fill((65, 105, 225))
        pygame.draw.lines(surface, color='RED', closed=False, points=self.vertices, width=self.widths)
        pygame.draw.lines(surface, color='WHITE', closed=False, points=self.vertices, width=1)

    def findClosestSegment(self, point: pygame.Vector2):
        p=np.array(point)
        ds=_lineseg_dists(p,self.aPointList,self.bPointList)
        pass


    def load(self,file):
        # https://stackoverflow.com/questions/15857818/python-svg-parser
        logger.info('loading track from {}'.format(file))
        self.paths, self.attributes = svg2paths(file)
        # for (p,a) in zip(paths,attributes):
        #     print('path='+str(p))
        #     print('attributes='+str(a))
        # self.drawing=svg2rlg(file)

        # compute bounding box
        xmin=float('inf')
        xmax=float('-inf')
        ymin=float('inf')
        ymax=float('-inf')
        for p in self.paths:
            self.vertices=list()
            for l in p:
                start=[l.start.real,l.start.imag]
                end=[l.end.real, l.end.imag]
                if start[0]<xmin: xmin=start[0]
                if end[0]<xmin: xmin=end[0]
                if start[0]>xmax: xmax=start[0]
                if end[0]>xmax: xmax=end[0]

                if start[1]<ymin: ymin=start[1]
                if end[1]<ymin: ymin=end[1]
                if start[1]>ymax: ymax=start[1]
                if end[1]>ymax: ymax=end[1]
                self.vertices.append(start)
                self.vertices.append(end)
            srcwidth=xmax-xmin
            srcheight=ymax-ymin
        # now we have points of track vertices, but they are not scaled to screen yet.
        size=.9
        scalex= SCREEN_WIDTH_PIXELS / float(srcwidth)
        scaley= SCREEN_HEIGHT_PIXELS / float(srcheight)
        # transform so that track fits into 90% of screen
        for p in self.vertices:
            p[0]= (1-size) * SCREEN_WIDTH_PIXELS + (p[0] - xmin) * (size - (1 - size)) * scalex
            p[1]= (1-size) * SCREEN_HEIGHT_PIXELS + (p[1] - ymin) * (size - (1 - size)) * scaley
        self.boundingBox=pygame.Rect(xmin,ymin,srcwidth,srcheight) # Rect(left, top, width, height) # todo in original units now
        alist=list()
        blist=list()
        ps=self.vertices
        n=len(ps)
        for i in range(n-1):
            alist.append(ps[i])
            blist.append(ps[i+1])
        self.aPointList=np.array(alist)
        self.bPointList=np.array(blist)

        pass


def _lineseg_dists(p:np.array, a:np.array, b:np.array):
    """Cartesian distances from points to line segments

https://stackoverflow.com/questions/27161533/find-the-shortest-distance-between-a-point-and-line-segments-not-line

    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892

    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2) # first point of each segment
        - b: np.array of shape (x, 2) # 2nd point of each segment

    returns np.array of distances to each segment

    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)


if __name__ == '__main__':
    track=Track('../media/testCourse.svg')
    track.findClosestSegment(pygame.Vector2(0, 0))


# Version of the track based on the extracting contour and loading png
class Track:
    def __init__(self, file='./media/track.png'):
        self.track = pygame.image.load(file)

    def draw(self, surface:pygame.surface):
        surface.fill((65, 105, 225))
        #surface.blit(self.track, (SCREEN_WIDTH//2 - car.car_state.position.x, SCREEN_HEIGHT//2 - car.car_state.position.y))
        surface.blit(self.track,(0,0))
