import logging

import pygame
import logging
# import svglib
# from svglib.svglib import svg2rlg
import cmath
import numpy as np
from svgpathtools import svg2paths
from src.globals import SCREEN_WIDTH_PIXELS, SCREEN_HEIGHT_PIXELS, M_PER_PIXEL
from timeit import default_timer as timer

logger = logging.getLogger(__name__)

def list_tracks():
    ''' :returns list of all available tracks as list(str)'''
    from os import listdir

    def list_files(directory, extension):
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    tr=list()
    directory = 'media/tracks'
    files = list_files(directory, "png")
    for f in files:
        tr.append(f.strip('.png'))
    return tr

# Version of the track based on the extracting contour and loading png
def get_position_on_map(car_state=None, x=None, y=None):
    if car_state is not None:
        x_map = int(car_state.position_m.x / M_PER_PIXEL)
        y_map = int(car_state.position_m.y / M_PER_PIXEL)
    elif (x is not None) and (y is not None):
        x_map = int(x / M_PER_PIXEL)
        y_map = int(y / M_PER_PIXEL)
    else:
        logger.info('To little data to calculate car_position. Return None.')
        return None

    return x_map, y_map


def closest_node(x, y, x_vector, y_vector):
    dist_2 = (x_vector - x) ** 2 + (y_vector - y) ** 2
    return np.argmin(dist_2)


class track:
    def __init__(self, track_name='track', media_folder_path='./media/tracks/'):
        self.name = track_name
        self.track_image = pygame.image.load(media_folder_path + track_name + '.png')
        self.track_map = np.load(media_folder_path + track_name + '_map.npy')
        self.TrackInfo = np.load(media_folder_path + track_name + '_info.npy', allow_pickle='TRUE').item()
        self.waypoints_x = self.TrackInfo['waypoint_x']
        self.waypoints_y = self.TrackInfo['waypoint_y']
        self.num_waypoints = len(self.waypoints_x)

        self.angle_next_segment_east = self.TrackInfo['AngleNextSegmentEast']
        self.angle_next_waypoint = self.TrackInfo['AngleNextCheckpointEast']

        if track_name == 'oval_easy':
            self.waypoints_search_radius = 160
            dy = 140
        else:
            self.waypoints_search_radius = 80
            dy = 70

        self.starting_line_rect = pygame.Rect(self.waypoints_x[0], self.waypoints_y[0]-dy, 100, 2*dy)
        # This is a pygame rectangle to automatically check if you passed the starting line
        # take for it +/-dy = 52 in map units
        self.anti_cheat_rect = pygame.Rect(self.waypoints_x[self.num_waypoints//2], self.waypoints_y[self.num_waypoints//2]-60, 120, 120)

        self.start_angle = self.angle_next_segment_east[0]
        if self.waypoints_y[0]>max(self.waypoints_y)/2:  # Remember y points down, Here I am down
            self.start_position_1 = np.array((self.waypoints_x[0], self.waypoints_y[0]+20))  # out
            self.start_position_2 = np.array((self.waypoints_x[0]+40, self.waypoints_y[0]-20))    # in
        else:  # Here I am up
            self.start_position_1 = np.array((self.waypoints_x[0], self.waypoints_y[0]-20))  # out
            self.start_position_2 = np.array((self.waypoints_x[0]-40, self.waypoints_y[0]+20))  # in

    def draw(self, surface: pygame.surface):
        surface.fill((65, 105, 225))
        # surface.blit(self.track, (SCREEN_WIDTH//2 - car.car_state.position.x, SCREEN_HEIGHT//2 - car.car_state.position.y))
        surface.blit(self.track_image, (0, 0))

    def get_surface_type(self, car_state=None, x=None, y=None):
        x, y = get_position_on_map(car_state=car_state, x=x, y=y)
        if x < 0 or x >= self.track_map.shape[1] or y < 0 or y > self.track_map.shape[0]:
            return 0
        return self.track_map[y, x]

    def get_nearest_waypoint_idx(self, car_state=None, x=None, y=None):
        x_map, y_map = get_position_on_map(car_state=car_state, x=x, y=y)

        # https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
        waypoints_idx_considered = np.where((self.waypoints_x > x_map - self.waypoints_search_radius) \
                                            & (self.waypoints_x < x_map + self.waypoints_search_radius) \
                                            & (self.waypoints_y > y_map - self.waypoints_search_radius) \
                                            & (self.waypoints_y < y_map + self.waypoints_search_radius))

        waypoints_idx_considered = np.atleast_1d(np.array(waypoints_idx_considered).squeeze())

        if len(waypoints_idx_considered) == 0:
            logger.warning('Error! No closest waypoints found!')
            return None

        idx = closest_node(x=x_map,
                           y=y_map,
                           x_vector=self.waypoints_x[waypoints_idx_considered],
                           y_vector=self.waypoints_y[waypoints_idx_considered])

        closest_waypoint = waypoints_idx_considered[idx]

        return closest_waypoint

    def get_current_angle_to_road(self, car_state=None,
                                  angle_car=None,
                                  x=None,
                                  y=None,
                                  nearest_waypoint_idx=None):

        x_map, y_map = get_position_on_map(car_state=car_state, x=x, y=y)

        if nearest_waypoint_idx is None:
            nearest_waypoint_idx = self.get_nearest_waypoint_idx(car_state=car_state, x=x_map, y=y_map)

        if angle_car is None:
            if car_state is not None:
                angle_car = car_state.body_angle_deg
            else:
                logger.warning("Error! Car angle for track.get_current_angle not provided")
                return None

        angle_car = angle_car - 360.0 * np.rint(angle_car / 360.0)

        angele_to_road = angle_car - self.angle_next_segment_east[nearest_waypoint_idx]

        # logger.info(angele_to_road)
        return angele_to_road

    def get_distance_to_nearest_segment(self,
                                        car_state=None,
                                        x_car=None,
                                        y_car=None,
                                        nearest_waypoint_idx=None):
        x_map, y_map = get_position_on_map(car_state=car_state, x=x_car, y=y_car)

        if nearest_waypoint_idx is None:
            nearest_waypoint_idx = self.get_nearest_waypoint_idx(car_state=car_state, x=x_map, y=y_map)

        p_car = np.array((x_map, y_map))
        if nearest_waypoint_idx == 0:
            p1 = np.array((self.waypoints_x[-1], self.waypoints_y[-1]))
        else:
            p1 = np.array((self.waypoints_x[nearest_waypoint_idx - 1], self.waypoints_y[nearest_waypoint_idx - 1]))

        if nearest_waypoint_idx == len(self.waypoints_x) - 1:
            p2 = np.array((self.waypoints_x[0], self.waypoints_y[0]))
        else:
            p2 = np.array((self.waypoints_x[nearest_waypoint_idx + 1], self.waypoints_y[nearest_waypoint_idx + 1]))

        d = np.linalg.norm(np.cross(p2 - p1, p1 - p_car)) / np.linalg.norm(p2 - p1)

        d = d * M_PER_PIXEL

        v1 = (p_car-p1)/np.linalg.norm(p_car-p1)
        v2 = (p2-p1)/np.linalg.norm(p2-p1)

        if v1[0] * v2[1] - v1[1] * v2[0] > 0:
            d = -d

        # logger.info(d)

        return d

    def car_completed_round(self, car_model):
        x_map = int(car_model.car_state.position_m.x / M_PER_PIXEL)
        y_map = int(car_model.car_state.position_m.y / M_PER_PIXEL)

        if self.anti_cheat_rect.collidepoint(x_map, y_map):
            car_model.passed_anti_cheat_rect = True

        if self.starting_line_rect.collidepoint(x_map, y_map):
            if car_model.passed_anti_cheat_rect:

                car_model.passed_anti_cheat_rect = False
                current_time = timer()
                car_model.car_state.time_results.append(current_time)
                if car_model.round_num == 0:
                    logger.info('Start!')
                elif car_model.round_num == 1:
                    logger.info('Completed ' + str(car_model.round_num) + ' round!')
                    s = 'Your time in the last round was  {:.2f}s'\
                        .format(current_time-car_model.car_state.time_results[car_model.round_num-1])
                    logger.info(s)
                else:
                    logger.info('Completed ' + str(car_model.round_num) + ' rounds!')
                    s = 'Your time in the last round was  {:.2f}s'\
                        .format(current_time-car_model.car_state.time_results[car_model.round_num-1])
                    logger.info(s)
                car_model.round_num += 1

        else:
            pass
