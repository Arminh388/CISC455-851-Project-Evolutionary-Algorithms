import random
import copy
from enum import Enum


# Runs a simple population initialization for an evolutionary algorithm
# (fitness evaluation and operators not implemented yet).
def fit_furniture(obstacles, furniture):
    w = len(obstacles)
    l = len(obstacles[0])
    n = len(furniture)

    POP_SIZE = 1
    pop = [Pos.rand(w, l) for _ in range(POP_SIZE)]

    # ADD EA below
    return pop


# Calculates the fitness (lower number is better).
def eval_fit(obstacles, furniture, pop):
    n = len(furniture)

    OOB_PENALTY = 3

    overlap = copy.deepcopy(obstacles)
    loss = 0

    # calculate out-of-bounds value and add to overlap
    for i in range(n):
        loss += furniture[i].project_to(overlap, pop[i])

    loss *= OOB_PENALTY

    # calc overlapping furniture
    overlap = [[max(e - 1, 0) for e in row] for row in overlap]
    loss += sum(sum(row) for row in overlap)

    return loss


class Furniture:
    def __init__(self, occupancy):
        self.occupancy = occupancy
        self.w = len(occupancy)
        self.l = len(occupancy[0])

    def project_to(self, grid, pos):
        w = len(grid)
        l = len(grid[0])
        dx = Pos.DELTA_X[pos.rot]
        dy = Pos.DELTA_Y[pos.rot]

        bounds = BoundingBox(Vec(0, 0), Vec(w, l))
        origin = pos.to_vec()

        oob = 0
        for x in range(self.w):
            for y in range(self.l):
                offset = Vec(x * dx, y * dy)
                offset.add(origin)

                if bounds.point_in_bounds(offset):
                    grid[offset.x][offset.y] += self.occupancy[x][y]
                else:
                    oob += self.occupancy[x][y]

        return oob


class Dir(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Pos:
    DELTA_X = {
        Dir.NORTH: 1,
        Dir.EAST: -1,
        Dir.SOUTH: -1,
        Dir.WEST: 1,
    }
    DELTA_Y = {
        Dir.NORTH: 1,
        Dir.EAST: 1,
        Dir.SOUTH: -1,
        Dir.WEST: -1,
    }

    def __init__(self, x, y, rot):
        self.x = x
        self.y = y
        self.rot = rot

    def to_vec(self):
        return Vec(self.x, self.y)

    @staticmethod
    def rand(w, l):
        x = random.randrange(w)
        y = random.randrange(l)
        rot = random.choice(list(Dir))
        return Pos(x, y, rot)


class BoundingBox:
    def __init__(self, v1, v2):
        # ensure v1 contains the smaller coordinates
        self.v1 = Vec(min(v1.x, v2.x), min(v1.y, v2.y))
        self.v2 = Vec(max(v1.x, v2.x), max(v1.y, v2.y))

    def point_in_bounds(self, v):
        if v.x < self.v1.x or v.x > self.v2.x:
            return False
        if v.y < self.v1.y or v.y > self.v2.y:
            return False
        return True


class Vec(list):
    def __init__(self, *args):
        # allow Vec([1,2,3]) or Vec(1,2,3)
        if len(args) == 1 and isinstance(args[0], (list, Vec)):
            super().__init__(args[0])
        else:
            super().__init__(args)

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, num):
        self[0] = num

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, num):
        self[1] = num

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, num):
        self[2] = num

    def add(self, *vs):
        max_len = max((len(v) for v in vs), default=0)
        for v in vs:
            for i, e in enumerate(v):
                if i < len(self):
                    self[i] += e
                else:
                    self.append(e)
        return self

    def subtract(self, v):
        max_len = max(len(self), len(v))
        for i in range(max_len):
            a = self[i] if i < len(self) else 0
            b = v[i] if i < len(v) else 0
            if i < len(self):
                self[i] = a - b
            else:
                self.append(-b)
        return self

    def scale(self, s):
        for i in range(len(self)):
            self[i] *= s
        return self

    def magn(self):
        import math
        return math.hypot(*self)

    def norm(self):
        mag = self.magn()
        if mag != 0:
            self.scale(1 / mag)
        return self

    def copy(self, v):
        self.clear()
        self.extend(v)
        return self

    def equals(self, obj):
        return isinstance(obj, Vec) and all(a == b for a, b in zip(self, obj))

    # static helpers mirroring original JS utilities
    @staticmethod
    def add_static(*vs):
        max_len = max((len(v) for v in vs), default=0)
        result = Vec([0] * max_len)
        for v in vs:
            for i, e in enumerate(v):
                result[i] += e
        return result

    @staticmethod
    def subtract_static(v1, v2):
        max_len = max(len(v1), len(v2))
        result = Vec([0] * max_len)
        for i in range(max_len):
            result[i] = (v1[i] if i < len(v1) else 0) - (v2[i] if i < len(v2) else 0)
        return result

    @staticmethod
    def scale_static(v, s):
        return Vec([e * s for e in v])

    @staticmethod
    def dot(v1, v2):
        max_len = max(len(v1), len(v2))
        return sum((v1[i] if i < len(v1) else 0) * (v2[i] if i < len(v2) else 0) for i in range(max_len))

    @staticmethod
    def cross(v1, v2):
        return Vec(
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )
