from typing import List, Tuple, Optional

import pygame
import numpy as np

from l2race_settings import M_PER_PIXEL, TRACKS_FOLDER, GAME_FONT_SIZE, GAME_FONT_NAME

# Functions for finding hit position
from scipy.special import tandg, cotdg, cosdg, sindg

from l2race_utils import my_logger

logger = my_logger(__name__)



def squared_distance(p1, p2):
    squared_distance = abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2
    return squared_distance



def list_tracks() -> List[str]:
    """list all available tracks as list(str)

    :returns: List of track names
    """
    from os import listdir

    def list_files(directory, extension):
        """
        Returns the list of files having extension given by "extension" argument
            and being included in directory given by "directory" argument
        :param directory: Directory which content should be listed
        :param extension: Only files with this extension will be included on the list
        :return: List of files in the directory with the predefined extension, with names in lower-case
        """
        return (f for f in listdir(directory) if f.endswith('.' + extension))

    # Create empty list to keep all the track names
    tr = list()
    # Give the name of the directory in which we want to look for the available tracks
    directory = 'media/tracks'
    # Find all the png files in the directory - these are the pictures of available tracks
    files = list_files(directory, "png")

    # For every found png file

    def rchop(s, suffix):
        if suffix and s.endswith(suffix):
            return s[:-len(suffix)]
        return s

    for f in files:
        # ...strip the .png suffix from its name to get track name and append to the list of all track names
        tr.append(rchop(f, '.png'))
    # Return track names list
    return tr


def get_position_on_map(car_state=None, x=None, y=None) -> Optional[Tuple[float, float]]:
    """
    The function converts the physical position (meters) to the position on the map (pixels).
    As the ..._map.npy is a discrete map the position is rounded down to the nearest integer after conversion.
    :param: One can provide either full car state (car_state), in which case the function will extract position from it,
            Or one can provide x-y coordinates of the car directly. The last option allows to convert to map units
            an arbitrary position, not only the instantaneous position of the car
    :return: position (x,y) of the car in map units (pixels) or None if not enough parameters were supplied
    """
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


def pixels2meters(x_map: float):
    """
    The function converts a value in the map units (pixels) to the physical units (meters).
    It is suitable to convert position, velocity or acceleration.
    :param x_map: value in map units (pixels, not necessarily integer)
    :return x_track: Value converted to physical units (meters)
    """
    x_track = x_map * M_PER_PIXEL
    return x_track


def meters2pixels(x_track: float):
    """
    The function converts a value in the map units (pixels) to the physical units (meters).
    In contrast to get_position_on_map() it DOES NOT round the results down to nearest integer.
    It is suitable to convert position, velocity or acceleration.
    :param x_track: Value converted to physical units (meters)
    :return x_map: Value in map units (pixels, not necessarily integer!)
    """
    x_map = x_track / M_PER_PIXEL
    return x_map


def closest_node(x, y, x_vector, y_vector):
    """
    The function finds the nearest target point, from target point list (normally a waypoints list)
    to a reference point (normally a car position)

    :param x: Position (its x and y coordinates) of the reference point (normally of the car) in map units (pixels)
    :param y: Position (its x and y coordinates) of the reference point (normally of the car) in map units (pixels)
    :param x_vector, y_vector: List of positions of target points in map units
    :return: Index of the nearest target point
    """
    dist_2 = (x_vector - x) ** 2 + (y_vector - y) ** 2
    return np.argmin(dist_2)


def get_neighbours(p_ref, ref_array):
    """
    Given a reference point p_ref = (i,j) (array cell) and the 2D array it is part of
    return the list of its neighbours (bu providing their rows and columns)
    :param p_ref: Tuple giving a row and column of the reference point p_ref
    :param ref_array: The 2D array p_ref is part of
    :return: The list of neighbours of p_ref in the form: [(i0,j0),(i1,j1),(i2,j2),...]
    """
    (i, j) = p_ref
    (h, w) = ref_array.shape

    neighbours = []

    if i > 0:
        neighbours.append((i - 1, j))
        if j > 0:
            neighbours.append((i - 1, j - 1))
        if j < (w - 1):
            neighbours.append((i - 1, j + 1))

    if i < (h - 1):
        neighbours.append((i + 1, j))
        if j > 0:
            neighbours.append((i + 1, j - 1))
        if j < (w - 1):
            neighbours.append((i + 1, j + 1))

    if j > 0:
        neighbours.append((i, j - 1))

    if j < (w - 1):
        neighbours.append((i, j + 1))

    return neighbours


def find_hit_position(angle, pos, track_map, dl=1.0):
    """
    This function returns the point at which
        a beam coming from point pos at angle given by angle variable
         for the first time crosses a non zero point on the map
         Timing at Marcin computer: dl = 5.0 -> about max 0.4 ms;
         It is roughly valid: dl/n -> time*n
     :param angle: angle (deg) in which the beam is going, e.g the body angle of the car
     :param pos: point ((x,y) in pixels) where the beam starts, e.g. position of the car
     :param track_map: a SPECIAL map which is non-zero in sand region and zero on track. Use map_lidar for it.
     :param dl: determines the precision of the collision point determination
                - the algorithm moves along the beam and checks every dl pixels (does not need to be integer)
                if it left the track
     :return the point on the track boundary which the beam first hits ((x,y) in pixels)
                    (e.g. which the car would hit if it would go on a straight line)
    """
    (h, w) = track_map.shape
    found = False

    # x = meters2pixels(pos[0])
    # y = meters2pixels(pos[1])

    x = pos[0]
    y = pos[1]

    # Depending on direction we take y = ax+b or x = ay+b
    if (45.0 < angle < 135.0) or (225.0 < angle < 315.0):
        # x = ay+b

        dy = dl * sindg(angle)
        # dy = dl
        a = cotdg(angle)
        b = x - a * y

        while 0 < x < w and 0 < y < h:
            if track_map[int(y), int(x)] > 0:
                found = True
                break
            else:
                y = y + dy
                x = a * y + b


    else:
        # dx = dl
        dx = dl * cosdg(angle)
        a = tandg(angle)
        b = y - a * x

        while 0 < x < w and 0 < y < h:
            if track_map[int(y), int(x)] > 0:
                found = True
                break
            else:
                x = x + dx
                y = a * x + b

    if found:
        hit_pos = (x, y)
    else:
        hit_pos = None

    return hit_pos


def calculate_distance(pos_1, pos_2):
    if pos_1 is not None and pos_2 is not None:
        d = np.linalg.norm(np.array(pos_1) - np.array(pos_2))
    else:
        d = 0
    return d

def get_surface_type_name_from_key(key:int) -> str:
    """
    :param key: the key for the surface type

    :returns string name of surface type
    """
    return track.SURFACE_TYPES[key] if key is not None else 'unknown'


class track:
    """
    The racetrack class. It has methods for finding wayoints.

    """
    CW = 'cw'
    CCW = 'ccw'
    # surface types
    WATER=0
    SAND=10
    SAND_LEFT=8
    SAND_RIGHT=12
    ASPHALT=20
    ASPHALT_LEFT=18
    ASPHALT_RIGHT=22
    CENTERLINE=30
    WAYPOINT=40

    SURFACE_TYPES = {WATER:'water',
                     SAND:'sand',
                     SAND_LEFT:'sand_left',
                     SAND_RIGHT:'sand_right',
                     ASPHALT: 'asphalt',
                     ASPHALT_LEFT: 'asphalt_left',
                     ASPHALT_RIGHT: 'asphalt_right',
                     CENTERLINE: 'centerline',
                     WAYPOINT: 'waypoint'
                     }


    def __init__(self, track_name=None, media_folder_path=TRACKS_FOLDER, waypoints_visible=1):
        """
        Constructs a new track instance.

        :param track_name: name of track without suffix, e.g. track_1
        :param media_folder_path: optional media folder path
        """
        if track_name is None: # no arg to use class variables like WATER
            return

        self.name = track_name
        logger.info('loading track image and info files with base name {}'.format(media_folder_path + track_name))
        try:
            png_ = media_folder_path + track_name + '.png'
            self.track_image = pygame.image.load(png_)
            map_npy_ = media_folder_path + track_name + '_map.npy'
            self.track_map = np.load(map_npy_, allow_pickle=True)
            info_npy_ = media_folder_path + track_name + '_info.npy'
            # see write_track_info_files.py for fields of TrackInfo dict
            self.TrackInfo = np.load(info_npy_,
                                     allow_pickle=True).item()  # fields are shows in write_track_info_files.py line 362 TODO document why .item() needed
        except FileNotFoundError as ex:
            logger.error(
                f'exception {ex} with filename {ex.filename2} trying to open track named {track_name} from folder {media_folder_path}')
            raise
        # TrackInfo dict has following
        # waypoint_x
        # waypoint_Y
        # DistNextCheckpointEast
        # AngleNextCheckpointEast
        # AngleNextCheckpointRelative TODO what is the difference betweeen checkpoint and waypoint?
        # AngleNextSegmentEast

        self.waypoints_x = self.TrackInfo['waypoint_x']  # Waypoint x coordinates in pixels
        self.waypoints_y = self.TrackInfo['waypoint_y']  # Waypoint y coordinates in pixels
        self.AngleDegNextCheckpointEast = M_PER_PIXEL * self.TrackInfo[
            'AngleNextCheckpointEast']  # absolute Angles of track line segments in degrees
        self.AngleDegNextCheckpointRelative = M_PER_PIXEL * self.TrackInfo[
            'AngleNextCheckpointRelative']  # Relative angles of track line segments in degrees
        self.waypoints = [[self.waypoints_x[i] * M_PER_PIXEL, self.waypoints_y[i] * M_PER_PIXEL] for i in
                          range(len(self.waypoints_x))]

        self.num_waypoints = len(self.waypoints_x)

        self.angle_next_segment_east_deg = self.TrackInfo[
            'AngleNextSegmentEast']  # angle is in degrees of the next segment relative to east which is 0 degrees
        self.angle_next_waypoint_deg = self.TrackInfo['AngleNextCheckpointEast']  # TODO ???? what is this?

        if track_name == 'oval_easy':
            self.waypoints_search_radius = 160
            dy = 140
        else:
            self.waypoints_search_radius = 80
            dy = 70

        # starting_line_rect defines the start line of the track. It is a vertical bar perpindicular to the the first waypoint of the track.
        # TODO define the starting line by a separate line in the track_info npy file
        self.starting_line_rect = pygame.Rect(self.waypoints_x[0], self.waypoints_y[0] - dy, 100, 2 * dy)
        # anti_cheat_rect is a pygame rectangle to automatically check if you passed the starting line
        # take for it +/-dy = 52 in map units
        self.anti_cheat_rect = pygame.Rect(self.waypoints_x[self.num_waypoints // 2],
                                           self.waypoints_y[self.num_waypoints // 2] - 60, 120, 120)

        logger.info(f'track direction is {self.TrackInfo["TrackDirection"]}')
        self.start_angle_deg = self.angle_next_segment_east_deg[0]# if self.is_clockwise() else self.angle_next_segment_east_deg[0] + 180
        # compute the possible starting positions of cars that will selected later from

        if self.waypoints_y[0] > max(self.waypoints_y) / 2:  # Remember y points down, Here I am down below halfway on track
            self.start_position_1 = np.array((self.waypoints_x[0], self.waypoints_y[0] + 20))  # out
            self.start_position_2 = np.array((self.waypoints_x[0] + 40, self.waypoints_y[0] - 20))  # in
        else:  # Here I am up
            self.start_position_1 = np.array((self.waypoints_x[0], self.waypoints_y[0] - 20))  # out
            self.start_position_2 = np.array((self.waypoints_x[0] - 40, self.waypoints_y[0] + 20))  # in

        # The following part plots waypoints on the track
        self.surface_waypoints = None
        if waypoints_visible is not None:
            self.create_waypoints_surface(waypoints_visible)

        self.map_lidar = np.copy(self.track_map)
        self.map_lidar[self.map_lidar != 10] = 0
        self.game_font=None # only initialized if we need to draw track

    def create_waypoints_surface(self, waypoints_visible):
        """
        Method that modifies the self.surface_waypoints to display red dots of predefines magnitude at the waypoints
        coordinates and be transparent at other coordinates
        :param waypoints_visible: If None this function will not be called and no waypoints are displayed.
        Otherwise waypoints_visible give the magnification of the waypoints in drawing (must be integer)
        :return: It does not return anything
        """
        map_waypoints = np.copy(self.track_map)
        map_waypoints[map_waypoints != 40] = 0
        map_waypoints[map_waypoints == 40] = 1
        # Make waypoints bigger to make them better visible
        for i in range(waypoints_visible):  # waypoints visible defines the magnification of the waypoints
            for p_ref in np.argwhere(map_waypoints == 1).tolist():
                neighbours = get_neighbours(p_ref, map_waypoints)
                for p in neighbours:
                    map_waypoints[p] = 1
        map_waypoints = np.stack((255 * map_waypoints, 0 * map_waypoints, 0 * map_waypoints,), axis=2)
        self.surface_waypoints = pygame.surfarray.make_surface(map_waypoints.transpose((1, 0, 2)))
        BLACK = (0, 0, 0)
        self.surface_waypoints.set_colorkey(BLACK)  # Black colors will not be blit.

    def draw(self, surface: pygame.surface):
        """
        Draw a track png
        :param surface: Pygame surface to which track png should be drawn
        """
        surface.fill((65, 105, 225))
        # surface.blit(self.track, (SCREEN_WIDTH//2 - car.car_state.position.x, SCREEN_HEIGHT//2 - car.car_state.position.y))
        surface.blit(self.track_image, (0, 0))
        if self.surface_waypoints is not None:
            surface.blit(self.surface_waypoints, (0, 0))
        if self.game_font is None:
            try:
                self.game_font = pygame.freetype.SysFont(GAME_FONT_NAME, GAME_FONT_SIZE)
            except:
                logger.warning('cannot get specified globals.py font {}, using pygame default font'.format(GAME_FONT_NAME))
                self.game_font = pygame.freetype.SysFont(pygame.font.get_default_font(), GAME_FONT_SIZE)
        self.game_font.render_to(surface, (0,0 + GAME_FONT_SIZE), self.name, (255,255,255)),

    def is_clockwise(self):
        return self.TrackInfo['TrackDirection'] == track.CW

    def get_surface_type_key(self, car_state=None, x=None, y=None):
        """
        This function returns the number corresponding to the surface type at the position of the car given in car_state
        OR at the position given by x,y arguments. The numbers maps to surface type according the following keys in track.SURFACE_TYPES:
            0 - water (out of track, now forward movement possible, possible to go back on reverse gear)
            10 - sand (8 - left and 12 - right boundary) (out of track, only slow movement possible)
            20 - asphalt (18 - left and 22 - right boundary) (normal car dynamics)
            30 - middle line (normal car dynamics)
            40 - waypoints (normal car dynamics)

        :param car_state: car_state from which coordinates of the point of interest (car position) can me extracted
        :param x: x-coordinate of point of interest in meter (usually the car position)
        :param y: y_car: y-coordinate of point of interest in meter (usually the car position)

        :return: int value corresponding to the surface type at the point of interest, e.g. track.WATER if in water
        """

        # Getting x,y coordinates of the point of interest in the map units (pixels)
        x, y = get_position_on_map(car_state=car_state, x=x, y=y)
        # Checking if the point of interest lays on the map.
        # If not return 0
        if x < 0 or x >= self.track_map.shape[1] or y < 0 or y > self.track_map.shape[0]:
            return 0
        # if yes check on track_map what kind of surface is at the point of interest
        return self.track_map[y, x]


    def get_nearest_waypoint_idx(self, car_state=None, x=None, y=None)->int:
        """
        Function returns the index of the nearest waypoint to the point of reference.
        The coordinates of the point of reference may be either read from car_state (car position),
        OR they can be provided as as a x,y coordinates
        :param car_state: car_state from which coordinates of the point of interest (car postion) can me extracted
        :param x: x-coordinate of point of reference in meter (usually the car position)
        :param y: y-coordinate of point of reference in meter (usually the car position)

        :return: index of closest waypoint
        """
        x_map, y_map = get_position_on_map(car_state=car_state, x=x, y=y)

        # https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
        waypoints_idx_considered = np.where((self.waypoints_x > x_map - self.waypoints_search_radius)
                                            & (self.waypoints_x < x_map + self.waypoints_search_radius)
                                            & (self.waypoints_y > y_map - self.waypoints_search_radius)
                                            & (self.waypoints_y < y_map + self.waypoints_search_radius))

        waypoints_idx_considered = np.atleast_1d(np.array(waypoints_idx_considered).squeeze())

        if len(waypoints_idx_considered) == 0:
            waypoints_idx_considered = range(len(self.waypoints_x))

        idx = closest_node(x=x_map,
                           y=y_map,
                           x_vector=self.waypoints_x[waypoints_idx_considered],
                           y_vector=self.waypoints_y[waypoints_idx_considered])

        closest_waypoint = waypoints_idx_considered[idx]

        return closest_waypoint

    def get_closest_index(self, p):

        min_distance = 100000
        waypoint_index = 0
        for i in range(len(self.waypoints_x)):
            waypoint = [self.waypoints_x[i], self.waypoints_y[i]]
            dist = squared_distance(p, waypoint)
            if (dist < min_distance):
                min_distance = dist
                waypoint_index = i
        return waypoint_index

    def get_current_angle_to_road(self, car_state=None,
                                  angle_car=None,
                                  x=None,
                                  y=None,
                                  nearest_waypoint_idx=None):
        """
        Returns the angle (in DEGREES) to the "nearest segment" of the road from the point of reference (usually the car).

        Def: nearest segment
           is a line segment connecting the next waypoint before and the next waypoint after the nearest waypoint

        Assuming the car goes clockwise (which is the orientation of our tracks)
            negative distance means that the driver goes counterclockwise (=to the left)

        One can either supply the car_state or x,y coordinates of the point of reference

        :param car_state: car_state from which coordinates of the point of reference (car postion) can me extracted.
        :param x: x-coordinate of point of reference (usually the car position).
        :param y: y-coordinate of point of reference (usually the car position).
        :param angle_car: Angle of car in degrees with east 0 and increasing clockwise.
        :param nearest_waypoint_idx: Optional. Index of the nearest waypoint.
                If not provided or None it will be calculated in this function;
                if previously calculated it may be provided to save calculation time

        :return: Angle in degrees to the nearest segment, positive to right TODO check
        """

        x_map, y_map = get_position_on_map(car_state=car_state, x=x, y=y)

        if nearest_waypoint_idx is None:
            nearest_waypoint_idx = self.get_nearest_waypoint_idx(car_state=car_state, x=x_map, y=y_map)

        if angle_car is None:
            if car_state is not None:
                angle_car = car_state.body_angle_deg
            else:
                logger.warning("Error! Car angle for track.get_current_angle not provided")
                return None

        # angle_car = angle_car - 360.0 * np.rint(angle_car / 360.0)
        # print('angle car: {}'.format(angle_car))
        # print('angle segment: {}'.format(self.angle_next_segment_east[nearest_waypoint_idx]))

        angle_to_road = angle_car - self.angle_next_segment_east_deg[nearest_waypoint_idx]
        angle_to_road = angle_to_road - 360.0 * np.rint(angle_to_road / 360.0)
        # print('angle_to_road: {}'.format(angle_to_road))
        # logger.info(angele_to_road)
        return angle_to_road

    def get_distance_to_nearest_segment(self,
                                        car_state=None,
                                        x_car=None,
                                        y_car=None,
                                        nearest_waypoint_idx=None):
        """
        Returns the signed distance to the "nearest segment" of the road from the point of reference (usually the car).

        Def: nearest segment
           is a line segment connecting the next waypoint before and the next waypoint after the nearest waypoint

        Assuming the car goes clockwise (which is the orientation of our tracks)
            negative distance means that the driver has this segment on his right-hand side (it is on the left side of the road)

        One can either supply the car_state or x,y coordinates of the point of reference

        :param car_state: car_state from which coordinates of the point of reference (car postion) can me extracted
        :param x_car: x-coordinate of point of reference (usually the car position)
        :param y_car: y-coordinate of point of reference (usually the car position)
        :param nearest_waypoint_idx: Optional. Index of the nearest waypoint.
                If not provided or None it will be calculated in this function;
                if previously calculated it may be provided to save calculation time
        :return: Signed distance from the nearest segment, positive to right TODO negative to left of track
        """

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

        v1 = (p_car - p1) / np.linalg.norm(p_car - p1)
        v2 = (p2 - p1) / np.linalg.norm(p2 - p1)

        if v1[0] * v2[1] - v1[1] * v2[0] > 0:
            d = -d

        return d

    def car_completed_round(self, car_model):
        """
        Determines if car crossed starting line and completed laps.
        If yes, saves the time result
            and provides a information about completed round which should be printed on the screen through server message
        This function can be called only on server as it requires car_model which is unavailable for client,
            and also not a part of the track class.

        :param car_model: the car model

        :returns: string: Information about completed round to be displayed on the screen.
        """
        s = None
        x_map = int(car_model.car_state.position_m.x / M_PER_PIXEL)
        y_map = int(car_model.car_state.position_m.y / M_PER_PIXEL)

        if self.anti_cheat_rect.collidepoint(x_map, y_map):
            car_model.passed_anti_cheat_rect = True

        if self.starting_line_rect.collidepoint(x_map, y_map):
            if car_model.passed_anti_cheat_rect:

                car_model.passed_anti_cheat_rect = False
                current_time_car = car_model.time
                car_model.car_state.time_results.append(current_time_car)
                if car_model.round_num == 0:
                    s = 'Start! '
                    logger.info(
                        'Car crossed start line and is starting lap at {}s ***************************************************'.format(
                            car_model.time))
                elif car_model.round_num == 1:
                    s0 = 'Car completed lap number {} at {}'.format(str(car_model.round_num), car_model.time)
                    logger.info(s0)
                    s1 = 'Your time in the last lap was  {:.2f}s' \
                        .format(current_time_car - car_model.car_state.time_results[car_model.round_num - 1])
                    logger.info(s1)
                    s = s0 + '\n' + s1
                else:
                    s0 = 'Car completed lap number {} at {}'.format(str(car_model.round_num), car_model.time)
                    logger.info(s0)
                    s1 = 'Your time in the last lap was  {:.2f}s' \
                        .format(current_time_car - car_model.car_state.time_results[car_model.round_num - 1])
                    logger.info(s1)
                    s = s0 + '\n' + s1
                car_model.round_num += 1

        else:
            pass

        return s

    def get_position_on_map(self, car_state=None, x=None, y=None):
        """
        The function converts the physical position (meters) to the position on the map (pixels).
        As the ..._map.npy is a discrete map the position is rounded down to the nearest integer after conversion.
        :param: One can provide either full car state (car_state), in which case the function will extract position from it,
                Or one can provide x-y coordinates of the car directly. The last option allows to convert to map units
                an arbitrary position, not only the instantaneous position of the car
        :return: position (x,y) of the car in map units (pixels) or None if not enough parameters were supplied
        """
        return get_position_on_map(car_state=car_state, x=x, y=y)

    def meters2pixels(self, x_track):
        """
        The function converts a value in the map units (pixels) to the physical units (meters).
        In contrast to get_position_on_map() it DOES NOT round the results down to nearest integer.
        It is suitable to convert position, velocity or acceleration.
        :param x_track: Value converted to physical units (meters)
        :return x_map: Value in map units (pixels, not necessarily integer!)
        """
        return meters2pixels(x_track=x_track)

    def pixels2meters(self, x_map):
        """
        The function converts a value in the map units (pixels) to the physical units (meters).
        It is suitable to convert position, velocity or acceleration.
        :param x_map: value in map units (pixels, not necessarily integer)
        :return x_track: Value converted to physical units (meters)
        """
        return pixels2meters(x_map=x_map)

    def find_hit_position(self, angle, pos, dl=1.0):
        """
        This function returns the point at which
            a beam coming from point pos at angle given by angle variable
             for the first time crosses a non zero point on the map
             Timing at Marcin computer: dl = 5.0 -> about max 0.4 ms;
             It is roughly valid: dl/n -> time*n
         :param angle: angle (deg) in which the beam is going, e.g the body angle of the car
         :param pos: point ((x,y) in pixels) where the beam starts, e.g. position of the car
         :param track_map: a SPECIAL map which is non-zero in sand region and zero on track. Use map_lidar for it.
         :param dl: determines the precision of the collision point determination
                    - the algorithm moves along the beam and checks every dl pixels (does not need to be integer)
                    if it left the track
         :return the point on the track boundary which the beam first hits ((x,y) in pixels)
                        (e.g. which the car would hit if it would go on a straight line)
        """
        track_map = self.map_lidar
        return find_hit_position(angle=angle, pos=pos, track_map=track_map, dl=dl)

    def get_hit_distance(self, angle=None, x_car=None, y_car=None, car_state=None, dl=1.0):
        """
        The functions calculate the distance (in meters) to the point at which the object starting at position x_car, y_car
        and moving in the direction given by angle would hit the track boundary.
        :param angle: angle at which the object is moving from starting point (in degrees)
        :param car_state: If angle or starting position not given, they can be extracted from car_state
        :param x_car: x coordinate of the starting position (in meters)
        :param y_car: y coordinate of the starting position (in meters)
        :param dl: Precision with which the "hit-point" should be determined (in pixels)
        :return: distance to the "hit-point" in meters
        """
        x_map, y_map = get_position_on_map(car_state=car_state, x=x_car, y=y_car)
        pos_map = (x_map, y_map)
        track_map = self.map_lidar
        if angle is None:
            angle = car_state.body_angle_deg
        hit_pos = find_hit_position(angle=angle, pos=pos_map, track_map=track_map, dl=dl)
        d_map = calculate_distance(pos_map, hit_pos)
        d = pixels2meters(d_map)
        return d
