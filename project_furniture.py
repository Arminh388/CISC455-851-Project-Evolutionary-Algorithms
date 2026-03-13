import random
import math
import copy
from enum import Enum


def fit_furniture(obstacles, furniture):
    """
    Runs the Evolutionary Algorithm (EA).

    Parameters:
        obstacles (list): 
            a 2D list where inner lists represent rows in the gridspace.
            Values of 0 are unoccupied cells, values of 1 are cells occupied by obstacles.
        furniture (list): 
            a list of Furniture objects.

    Returns:
        pop (list): 
            a 2D list where inner lists are solutions represented by positions (Pos) for each Furniture object in the furniture parameter.
            Represents the final population after running EA.

    """
    l = len(obstacles)
    w = len(obstacles[0])
    n = len(furniture)

    POP_SIZE = 1

    # individuals are lists of positions for each furniture
    pop = [[Pos.rand(l, w) for _ in range(n)]
           for _ in range(POP_SIZE)]

    # ADD EA below
    return pop

def visualize_solution(obstacles, furniture, individual):
    """
    Prints a visualization of an individual to the terminal.

    Parameters:
        obstacles (list):
            a 2D list where inner lists represent rows in the gridspace.
            Values of 0 are unoccupied cells, values of 1 are cells occupied by obstacles.
        furniture (list):
            a list of Furniture objects.
        individual (list):
            a potential solution represented by positions (Pos) for each Furniture object in the furniture parameter.
    """
    
    l = len(obstacles)
    w = len(obstacles[0])
    n = len(furniture)

    grid = [[0 for _ in range(w)] for _ in range(l)]

    for i in range(n):
        furn = furniture[i] # furniture object
        pos = individual[i] # furniture's position
        
        furn.project_to(grid,pos) # place furniture

    # print grid
    header_txt = "┏  " + "  ".join(str(i) for i in range(w)) + "  ┓"
    print(header_txt)
    legend_txt = " Legend: . = empty, # = occupied, ■ = overlap"
    for x in range(l):
        row_text = f"{x}  " + "  ".join("#" if cell == 1 else ("■" if cell > 1 else ".") for cell in grid[x]) + f"  {x}"
        if x == l//2:
            row_text += legend_txt
        print(row_text)
    footer_text = "┗  " + "  ".join(str(i) for i in range(w)) + "  ┛"
    print(footer_text)


def eval_fit(obstacles, furniture, pop):
    """
    Calculates fitness of individuals in the population.
    Lower value => better fitness.

    Parameters:
        obstacles (list):
            a 2D list where inner lists represent rows in the gridspace.
            Values of 0 are unoccupied cells, values of 1 are cells occupied by obstacles.
        furniture (list): 
            a list of Furniture objects.
        pop (list): 
            a 2D list where inner lists are solutions represented by positions for each Furniture object in the furniture parameter.

    Returns:
        loss (int): 
            computed penalty for out-of-bounds and overlap with other furniture

    """

    n = len(furniture)

    OOB_PENALTY = 3

    overlap = copy.deepcopy(obstacles)
    loss = 0

    # calculate out-of-bounds penalty
    for i in range(n):
        pos = pop[i]
        loss += furniture[i].project_to(overlap, pos)

    loss *= OOB_PENALTY

    # calculate overlap
    overlap = [[max(e - 1, 0) for e in row] for row in overlap]
    loss += sum(sum(row) for row in overlap)

    return loss


# holds furniture shape
class Furniture:
    """
    Holds the furniture shape.

    Parameters:
        occupancy (list): 
            a 2D list where inner lists represent the rows of the rectangle enclosing the furniture. 
            Values of 1 in the rows indicate cells occupied by the furniture; values of 0 are cells that are unoccupied.

    Attributes:
        occupancy (list): 
            a 2D list where inner lists represent the rows of the rectangle enclosing the furniture. 
            Values of 1 in the rows indicate cellsoccupied by the furniture; values of 0 are cells that are unoccupied.
        l (int): 
            length (y) of rectangle enclosing furniture.
        w (int): 
            width (x) of rectangle enclosing furniture.
    """

    def __init__(self, occupancy):
        self.occupancy = occupancy
        self.l = len(occupancy)
        self.w = len(occupancy[0])

    def rotate(self, x, y, rot):
        """
        Rotates a coordinate of a furniture object.

        Parameters:
            x (int):
                the x-coordinate of the furniture object
            y (int):
                the y-coordinate of the furniture object
            rot (Pos.rot):
                the rotation to apply to the furniture

        Returns:
            (tuple):
                the rotated (x,y) coordinate
        
        """

        # rotation transformations
        ROT_MATRIX = {
            Pos.DIR.CW0: lambda x,y: (  x  ,   y  ),
            Pos.DIR.CW90:  lambda x,y: (self.l-1-y,   x  ),
            Pos.DIR.CW180: lambda x,y: (self.w-1-x, self.l-1-y),
            Pos.DIR.CW270:  lambda x,y: (  y  , self.w-1-x)   
        }

        return ROT_MATRIX[rot](x,y)

    def project_to(self, grid, pos):
        """
        Projects the furniture object onto a grid.

        Parameters:
            grid (list):
                a 2D list where inner lists represent rows of the grid. Values in the grid represent the number of furniture
                objects occupying the cell.
            pos (Pos):
                the position to project the furniture object to.

        Returns:
            oob (int):
                the number of cells occupied by the furniture that fall out-of-bounds.
        
        """

        l = len(grid)
        w = len(grid[0])


        bounds = BoundingBox(Vec(0, 0), Vec(w-1, l-1))
        origin = pos.to_vec()

        oob = 0

        for x in range(self.w):
            for y in range(self.l):

                rx, ry = self.rotate(x, y, pos.rot)

                offset = Vec(rx, ry)
                offset.add(origin)

                if bounds.point_in_bounds(offset):
                    grid[offset.y][offset.x] += self.occupancy[y][x]
                else:
                    oob += self.occupancy[y][x]

        return oob


# position with rotation
class Pos:
    """
    Represents the (x,y) coordinate and rotation for placing a furniture object.
    The (x,y) coordinate represents the top-left corner of the enclosing rectangle of the furniture.

    Parameters:
        x (int):
            the x-coordinate of the position. (DEFAULT = 0)
        y (int):
            the y-coordinate of the position. (DEFAULT = 0)
        rot (Pos.DIR):
            the rotation of the furniture. (DEFAULT = 0 = CW0)

    Attributes:
        x (int):
            the x-coordinate of the position.
        y (int):
            the y-coordinate of the position.
        rot (Pos.DIR):
            the rotation of the furniture.
    """

    class DIR(Enum):
        """
        Available furniture rotations.
        """
        CW0 = 0   # clockwise   0 degrees
        CW90 = 1  # clockwise  90 degrees
        CW180 = 2 # clockwise 180 degrees
        CW270 = 3 # clockwise 270 degrees

        # counter-clockwise aliases
        CCW0 = 0   # counter-clockwise   0 degrees
        CCW270 = 1 # counter-clockwise 270 degrees
        CCW180 = 2 # counter-clockwise 180 degrees
        CCW90 = 3  # counter-clockwise  90 degrees

        # NESW aliases
        NORTH = 0
        EAST = 1
        SOUTH = 2
        WEST = 3
    
    dir_to_str = {
        DIR.CW0 : "0° Clockwise",
        DIR.CW90 : "90° Clockwise",
        DIR.CW180 : "180° Clockwise",
        DIR.CW270 : "270° Clockwise"
    }

    def __init__(self, x=0, y=0, rot=0):
        self.x = x
        self.y = y
        self.rot = rot

    def to_vec(self):
        """
        Converts x, y coordinates to Vec object.
        """
        return Vec(self.x, self.y)

    @staticmethod
    def rand(l, w):
        """
        Randomizes a position object.

        Parameters:
            l (int):
                the length (y) of the gridspace on which the position resides.
            w (int):
                the width (y) of the gridspace on which the position resides.

        Returns:
            (Pos):
                a random Pos object.
        """

        x = random.randrange(w)
        y = random.randrange(l)

        rot = random.choice(list(Pos.DIR))

        return Pos(x, y, rot)


# bounding box
class BoundingBox:
    """
    Defines grid boundaries.

    Parameters:
        v1 (Vec):
            the lower-bound corner of the grid.
        v2 (Vec):
            the upper-bound corner of the grid.
    
    Attributes:
        v1 (Vec):
            the lower-bound corner of the grid.
        v2 (Vec):
            the upper-bound corner of the grid.
    """

    def __init__(self, v1, v2):

        self.v1 = v1
        self.v2 = v2

        if self.v1.x > self.v2.x:
            self.v1.x, self.v2.x = self.v2.x, self.v1.x

        if self.v1.y > self.v2.y:
            self.v1.y, self.v2.y = self.v2.y, self.v1.y

    def point_in_bounds(self, v):
        """
        Checks if a Vec object is within the boundaries of the grid.

        Parameters:
            v (Vec):
                the vector to be checked.

        Returns:
            (Bool):
                True if the vector is within the grid boundaries.
        """

        if v.x < self.v1.x or v.x > self.v2.x:
            return False

        if v.y < self.v1.y or v.y > self.v2.y:
            return False

        return True

# Armin's translation of Paul's Vec class
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



######################
# TEST OF VISUALIZER
######################

# empty 10x10 grid
obstacles = [[0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0]]

# 4 different furniture items
furnitures = [
    Furniture([[1,1]]),
    Furniture([[0,1],
               [1,1]]),
    Furniture([[1,1,1],
               [1,0,1],
               [1,0,0]]),
    Furniture([[1,1],
               [1,0],
               [1,0],
               [1,0]])]

# random positions for each furniture
rand_sol =[Pos.rand(len(obstacles), len(obstacles[0])) for _ in range(len(furnitures))]

visualize_solution(obstacles, furnitures, rand_sol)

######################
# SHOWCASE OF ROTATION
######################
x = 3
y = 4
print(f"\n======================\nROTATION OF FURNITURE\nTop-Left Corner @ ({x},{y})")

# furniture is shaped:
#     #  #  #  #  #
#     #  .  .  .  .
#     #  .  .  .  .
furnitures = [Furniture([[1,1,1,1,1],
                         [1,0,0,0,0],
                         [1,0,0,0,0]])]

for rot in Pos.DIR:
    print(f"\n{Pos.dir_to_str[rot]}")
    visualize_solution(obstacles,furnitures,[Pos(x,y,rot)])