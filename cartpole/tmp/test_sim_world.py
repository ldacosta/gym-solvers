import unittest

from random import random, randint
from cartpole.tmp.sim_world import SimWorld, Direction, angle_between


class TestBitCondition(unittest.TestCase):

    def setUp(self):
        pass

    def test_all_in_world(self):
        """Are agent and ball in world, all the time?"""
        for _ in range(100):
            world = SimWorld(size=randint(2,100), toroidal=random() > 0.5)
            self.assertTrue((world.ball_pos[0] >= 0) and (world.ball_pos[0] <= world.size - 1), "world.ball_pos = {}".format(world.ball_pos))
            self.assertTrue((world.ball_pos[1] >= 0) and (world.ball_pos[1] <= world.size - 1), "world.ball_pos = {}".format(world.ball_pos))
            world.place_agent(avoid_ball_position=random() > 0.5)
            self.assertTrue((world.agent_pos[0] >= 0) and (world.agent_pos[0] <= world.size - 1), "world.agent_pos = {}".format(world.agent_pos))
            self.assertTrue((world.agent_pos[1] >= 0) and (world.agent_pos[1] <= world.size - 1), "world.agent_pos = {}".format(world.agent_pos))
            for _ in range(world.size * world.size * 4):
                r = random()
                if r < 0.25:
                    d = Direction.RIGHT
                elif r < 0.5:
                    d = Direction.LEFT
                elif r < 0.75:
                    d = Direction.TOP
                else:
                    d = Direction.BOTTOM
                world.move_agent(d)
                self.assertTrue((world.agent_pos[0] >= 0) and (world.agent_pos[0] <= world.size - 1), "world.agent_pos = {}".format(world.agent_pos))
                self.assertTrue((world.agent_pos[1] >= 0) and (world.agent_pos[1] <= world.size - 1), "world.agent_pos = {}".format(world.agent_pos))

    def test_angle(self):
        """Angle 'between 2 points'"""
        a = angle_between((3,4), (0,3))
        # self.assertTrue((a > 90) and (a < 180))

    def test_distance(self):
        world = SimWorld(size=5, toroidal=True)
        pair_of_pts = [((1,2), (4,4)), ((4,0), (0,3))]
        for (pt1, pt2) in pair_of_pts:
            dx,dy = world._dist_pt2pt(pt1, pt2)
            the_d = world._euclidean_distance(pt1=(1,2), pt2=(4,4))
            print("{} -> {}: dx = {}, dy = {}, dist = {}".format(pt1, pt2, dx, dy, the_d))

    def test_move(self):
        # world = SimWorld(size=randint(2, 100), toroidal=random() > 0.5)

        for _ in range(10):
            world = SimWorld(size=randint(2,100), toroidal=True)
            # if I choose any direction, and move consistently that way,
            # I should get back to my initial position after 'size' steps:
            for _ in range(min(pow(world.size,2), 100)):
                world.place_agent(avoid_ball_position=False)
                orig_agent_pos = world.agent_pos
                for direction in Direction:
                    for step in range(world.size):
                        world.move_agent(direction)
                    self.assertEqual(world.agent_pos, orig_agent_pos, "World size = %d" % (world.size))
            # if I am in any location, RIGHT and LEFT are opposites, and TOP and BOTTOM:
            for _ in range(min(pow(world.size,2), 100)):
                world.place_agent(avoid_ball_position=False)
                orig_agent_pos = world.agent_pos
                world.move_agent(Direction.RIGHT)
                world.move_agent(Direction.LEFT)
                self.assertEqual(world.agent_pos, orig_agent_pos, "World size = %d" % (world.size))
                orig_agent_pos = world.agent_pos
                world.move_agent(Direction.TOP)
                world.move_agent(Direction.BOTTOM)
                self.assertEqual(world.agent_pos, orig_agent_pos, "World size = %d" % (world.size))

    def test_max_distance(self):
        """is max distance well calculated?"""
        for _ in range(10):
            world = SimWorld(size=randint(2,25), toroidal=True)  # TODO: put a random() > 0.5 here
            max_d = -1
            for x1 in range(world.size):
                for y1 in range(world.size):
                    for x2 in range(world.size):
                        for y2 in range(world.size):
                            d = world._euclidean_distance((x1,y1), (x2,y2))
                            if d > max_d:
                                max_d = d
            self.assertEqual(max_d, world.max_distance)

    def test_bar(self):
        """whatev."""
        world = SimWorld(size=15, toroidal=True)
        world.place_ball(pos=(6,5))
        world.place_agent_at(pos=(7,5))
        d1 = world.dist_agent_to_ball()
        a1 = world.angle_agent_to_ball()
        world.place_agent_at(pos=(5,5))
        d2 = world.dist_agent_to_ball()
        a2 = world.angle_agent_to_ball()
        print("d1 = %.2f, angle1: %.2f" % (d1, a1))
        print("d2 = %.2f, angle2: %.2f" % (d2, a2))






