import math

from random import randint
from typing import Tuple, Optional
from enum import Enum, auto

from xcs.bitstrings import BitString

import numpy as np

def angle_between(p1, p2) -> float:
    """Returns angle in degrees, so in [0, 360]"""
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang2 - ang1) % (2 * np.pi))


import numpy as np
from typing import Tuple
from random import randint

def angle_with_x(a_vector: Tuple[int, int]) -> float:
    """In degrees."""
    a = np.array([0 + 0j])
    a.imag = np.array([a_vector[1]])
    a.real = np.array([a_vector[0]])
    return np.angle(a, deg=True)[0]


class Direction(Enum):
    RIGHT = auto()
    LEFT = auto()
    TOP = auto()
    BOTTOM = auto()


class SimWorld(object):

    def _random_position(self) -> Tuple[int, int]:
        return randint(0, self.size - 1), randint(0, self.size - 1)

    def __init__(self, size: int, toroidal: bool):
        assert size > 0
        self.size = size
        self.is_toroid = toroidal
        self.ball_pos = self._random_position()
        self.agent_pos = None  # tuple: position of agent
        # max distance between 2 objects in this world.
        if not self.is_toroid:
            self.max_distance = self._euclidean_distance(pt1=(0,0), pt2=(self.size - 1, self.size - 1))
        else:
            max_dist_on_axe = int(math.floor(self.size/2))
            self.max_distance = self._euclidean_distance(pt1=(0,0), pt2=(max_dist_on_axe, max_dist_on_axe))

    def place_ball(self, pos: Optional[Tuple[int, int]] = None):
        """Places ball at a (random) position in the grid"""
        if pos is None:
            self.ball_pos = self._random_position()
        else:
            assert (pos[0] >= 0) and (pos[0] < self.size) and (pos[1] >= 0) and (pos[1] < self.size)
            self.ball_pos = pos

    def place_agent_at(self, pos: Optional[Tuple[int, int]] = None):
        """Places agent at a random position in the grid, avoiding the ball position (depending on flag)"""
        if pos is None:
            self.place_agent(avoid_ball_position=True)
        else:
            assert (pos[0] >= 0) and (pos[0] < self.size) and (pos[1] >= 0) and (pos[1] < self.size)
            self.agent_pos = pos

    def place_agent(self, avoid_ball_position: bool = True):
        """Places agent at a random position in the grid, avoiding the ball position (depending on flag)"""
        assert (not avoid_ball_position) or (self.size > 1), "Can't avoid ball in world size = 1"
        self.agent_pos = self._random_position()
        while avoid_ball_position and (self.agent_pos == self.ball_pos):
            self.agent_pos = self._random_position()

    def _new_pos_if_moved(self, current_pos: Tuple[int,int], direction: Direction) -> Tuple[int, int]:
        if direction == Direction.RIGHT:
            delta_x, delta_y = 1, 0
        elif direction == Direction.LEFT:
            delta_x, delta_y = -1, 0
        elif direction == Direction.TOP:
            delta_x, delta_y = 0, -1
        elif direction == Direction.BOTTOM:
            delta_x, delta_y = 0, 1
        else:
            raise RuntimeError("canard!")
        if self.is_toroid:
            return (current_pos[0] + delta_x) % self.size, (current_pos[1] + delta_y) % self.size
        else:
            return max(0, min(current_pos[0] + delta_x, self.size - 1)), max(0, min((current_pos[1] + delta_y, self.size - 1)))

    def _agent_pos_if_moved(self, direction: Direction) -> Tuple[int, int]:
        return self._new_pos_if_moved(current_pos = self.agent_pos, direction = direction)

    def distance_to_ball_if_moved(self, from_pt: Tuple[int, int], direction: Direction) -> float:
        return self.dist_to_ball(pt=self._new_pos_if_moved(from_pt, direction))

    def agent_distance_to_ball_if_moved(self, direction: Direction) -> float:
        return self.distance_to_ball_if_moved(from_pt=self.agent_pos, direction=direction)

    def move_agent(self, direction: Direction) -> bool:
        new_pos = self._agent_pos_if_moved(direction)
        old_pos = self.agent_pos
        self.agent_pos = new_pos
        # print("Moved from {}, direction {} => new pos = {}".format(old_pos, direction.name, new_pos))
        return not (old_pos == self.agent_pos)

    def ball_at(self, pos: Tuple[int, int]) -> bool:
        return self.ball_pos == pos

    def _dist_pt2pt(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> (float, float):
        """Returns (dist_x, dist_y), with signs: - means 'go left' and + 'go right'"""

        def correct_delta(d: float) -> float:
            if abs(d) > (self.size / 2):
                return self.size + d if d < 0 else d - self.size
            else:
                return d

        assert (pt1[0] >= 0) and (pt1[0] < self.size) and (pt2[1] >= 0) and (pt2[1] < self.size)
        if not self.is_toroid:
            # dist_x = min((pt[0] - self.ball_pos[0]) % self.size, (self.ball_pos[0] - pt[0]) % self.size)
            # dist_y = min((pt[1] - self.ball_pos[1]) % self.size, (self.ball_pos[1] - pt[1]) % self.size)
            raise NotImplementedError()
        else:
            dist_x = correct_delta(pt2[0] - pt1[0])
            dist_y = correct_delta(pt2[1] - pt1[1])
            return dist_x, dist_y

    def _euclidean_distance(self, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
        dist_x, dist_y = self._dist_pt2pt(pt1=pt1, pt2=pt2)
        r = math.sqrt(math.pow(dist_x, 2) + math.pow(dist_y, 2))
        return r

    def dist_to_ball(self, pt: Tuple[int, int]) -> float:
        return self._euclidean_distance(pt1=pt, pt2=self.ball_pos)

    def dist_agent_to_ball(self) -> float:
        return self.dist_to_ball(self.agent_pos)

    def angle_to_ball(self, pt: Tuple[int, int]) -> float:
        if not self.is_toroid:
            a = angle_with_x(a_vector=(self.ball_pos[0] - pt[0], self.ball_pos[1] - pt[1]))
            raise NotImplementedError()
        else:
            dist_x, dist_y = self._dist_pt2pt(pt1=pt, pt2=self.ball_pos)
            the_vector = (dist_x, dist_y)
            aa = angle_with_x(the_vector)
            # a = angle_with_x(a_vector=((self.ball_pos[0] + dist_x) - pt[0], (self.ball_pos[1] + dist_y) - pt[1]))
        return aa

    def angle_agent_to_ball(self) -> float:
        return self.angle_to_ball(self.agent_pos)

    def agent_at_ball_position(self) -> bool:
        return self.agent_pos == self.ball_pos

    def agent_sensing(self) -> BitString:
        """
        Encodes all world that agent sees, by looking "in front" of it.
        Tries to see a world of dimension size x size, centered on its position.
        If the world is not toroidal then its vision stops at the end of the world.
        For each cell it sees it puts a 1 if a ball is there, 0 otherwise.
        :return: a BitString
        """
        if self.is_toroid:
            # give encoding of the 8 cells that surround me: from North to Northwest, clock-wise
            cells_to_sample = [
                (self.agent_pos[0], (self.agent_pos[1] - 1) % self.size),
                ((self.agent_pos[0] + 1) % self.size, (self.agent_pos[1] - 1) % self.size),
                ((self.agent_pos[0] + 1) % self.size, self.agent_pos[1]),
                ((self.agent_pos[0] + 1) % self.size, (self.agent_pos[1] + 1) % self.size),
                (self.agent_pos[0], (self.agent_pos[1] + 1) % self.size),
                ((self.agent_pos[0] - 1) % self.size, (self.agent_pos[1] + 1) % self.size),
                ((self.agent_pos[0] - 1) % self.size, self.agent_pos[1]),
                ((self.agent_pos[0] - 1) % self.size, (self.agent_pos[1] - 1) % self.size),
            ]
            b = BitString('')
            for x,y in cells_to_sample:
                b += BitString('1') if self.ball_at((x, y)) else BitString('0')
            return b

            # super-simplified version: 2-bits:
            # 1 bit => 0 if ball at my left, 1 if ball at my right (or right in front of me)
            # 1 bit => 0 if ball below me, 1 if ball ahead of me (or at same level).
            at_left = (self.agent_pos[0] - self.ball_pos[0] > 0)
            at_bottom = (self.agent_pos[1] - self.ball_pos[1] > 0)
            b = BitString('%d%d' % (at_left, at_bottom))
            return b
        else:
            raise NotImplementedError("dafuq!")
        if self.is_toroid:
            half_size = (self.size - 1) / 2
            data_before, data_after = math.ceil(half_size), math.floor(half_size)
            rows_raw = list(range(self.agent_pos[1] - data_before, self.agent_pos[1] + data_after + 1))
            rows = list(map(lambda v: v % self.size, rows_raw))
            cols_raw = list(range(self.agent_pos[0] - data_before, self.agent_pos[0] + data_after + 1))
            cols = list(map(lambda v: v % self.size, cols_raw))
            b = BitString('')
            for row in rows:
                for col in cols:
                    b += BitString('1') if self.ball_at((row, col)) else BitString('0')
            return b
        else:
            raise NotImplementedError("dafuq!")

