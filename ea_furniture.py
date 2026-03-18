import random
import math
import copy
import operator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np
from enum import Enum

def fit_furniture(obstacles, furniture, map_name=None, fit_thresh=np.inf, animate=False, open_reward=True, corner_reward=True, coverage_reward=False, reprod_fit=False, const_mut=True, immigration=True):
    """
    Runs the Evolutionary Algorithm (EA). Plots the fitness trajectory of evolution.

    Parameters:
        obstacles (list): 
            a 2D list where inner lists represent rows in the gridspace.
            Values of 0 are unoccupied cells, values of 1 are cells occupied by obstacles.
        furniture (list): 
            a list of Furniture objects.
        map_name (str):
            the name of the map used, if chosen from the pre-made list. (DEFAULT = None)
        fit_thresh (float):
            a fitness threshold to stop generations when the best member has fitness >= this value. 
            Default is infinity, meaning runs will always go until the GENERATION_COUNT. (DEFAULT = np.inf)
        animate (Bool):
            if True, plot the best and worst individual from certain generations as evolution progresses. (DEFAULT = False)
        open_reward (Bool):
            whether or not to add reward for adjacent empty cells. (DEFAULT = True)
        corner_reward (Bool):
            whether or not to add reward for furniture-occupied corner cells. (DEFAULT = True)
        coverage_reward (Bool):
            whether or not to add reward for furniture-occupied cells. (DEFAULT = False)
        reprod_fit (Bool):
            reproductive fitness. whether the offspring's chance to inherit a trait from each parent is proportional that parent's fitness. 
            If False, chance will be 50/50 (DEFAULT=False)
        const_mut (Bool):
            whether mutation rates remain constant throughout evolution. Rates will decrease with generations if set to False. (DEFAULT = True)
        immigration (Bool):
            whether randomly generated individuals should be added (with replacement) to the population at set intervals. (DEFAULT = True)
            
    Returns:
        pop (list): 
            a 2D list where inner lists are solutions represented by positions (Pos) for each Furniture object in the furniture parameter.
            Represents the final population after running EA.
        fitness (list):
            a list representing the fitness values of each individual in the population.
        gen (int):
            the final generation number reached.
    """
    l = len(obstacles)
    w = len(obstacles[0])
    n = len(furniture)

    POP_SIZE = 1000
    MATING_POOL_SIZE = 250
    TOURNAMENT_SIZE = 4
    GENERATION_COUNT = 1000
    SWAP_SCALE = 50             # lower = higher mutation rate
    XY_SCALE = 1                # higher = higher mutation rate
    ROT_SCALE = 1               # higher = higher mutation rate
    ANIMATION_CYCLE = 100       # plot best individual every <ANIMATION_CYCLE> generations
    IMMIGRANT_POP_SIZE = 50     # add this many immigrants every immigration cycle
    IMMIGRATION_CYCLE = 50      # introduce random individuals every <IMMIGRATION_CYCLE> generations
    MUTATION_CHANGE = 0.999     # should lie on the interval (0,1]. Smaller value means mutation rate depreciates faster

    best_fits = []
    avg_fits = []
    worst_fits = []
    mut_rates = []

    swap_chance = SWAP_SCALE
    xy_sigma = XY_SCALE
    rot_sigma = ROT_SCALE

    # individuals are lists of positions for each furniture
    pop = [[Pos.rand(l, w) for _ in range(n)]
           for _ in range(POP_SIZE)]
    
    # calculate individual fitnesses
    fitness = [eval_fit(obstacles, furniture, ind, open_reward, corner_reward, coverage_reward)[0] for ind in pop]
    
    gen = 0
    best_fit = -np.inf

    # run EA
    while (gen < GENERATION_COUNT + 1) and (best_fit < fit_thresh):

        if not const_mut:
            # mutation rates decrease with generations
            swap_chance = int(SWAP_SCALE*((1-MUTATION_CHANGE)**gen))
            xy_sigma = XY_SCALE*(MUTATION_CHANGE**gen)
            rot_sigma = ROT_SCALE*(MUTATION_CHANGE**gen)

        # get mating pairs
        mating_pool = tournament(pop, fitness, MATING_POOL_SIZE, TOURNAMENT_SIZE)

        # generate offspring from each mating pair
        offspring = []
        worst_fit = min(fitness)
        for p1, p2, p1_fit, p2_fit in mating_pool:
            p1_window = p1_fit - worst_fit
            p2_window = p2_fit - worst_fit
            offspring.append(crossover(p1, p1_window, p2, p2_window, reprod_fit))
        
        # mutate offspring
        for ind in offspring:
            mutate(ind, swap_chance, xy_sigma, rot_sigma)
        
        # calculate offspring fitness
        offspring_fitness = [eval_fit(obstacles, furniture, ind, open_reward, corner_reward, coverage_reward)[0] 
                             for ind in offspring]

        if immigration:
            # create random immigrants, add them to offspring pool
            if gen % IMMIGRATION_CYCLE == 0:
                immigrants = [[Pos.rand(l,w) for _ in range(n)] 
                            for _ in range(IMMIGRANT_POP_SIZE)]
                immigrant_fitness = [eval_fit(obstacles, furniture, ind, open_reward, corner_reward, coverage_reward)[0] for ind in immigrants]
                offspring += immigrants
                offspring_fitness += immigrant_fitness

        # replace population
        pop, fitness = replacement(pop, fitness, offspring, offspring_fitness)

        # generation stats
        best_fit = max(fitness)
        worst_fit = min(fitness)
        avg_fit = sum(fitness)/len(fitness)
        best_fits.append(best_fit)
        worst_fits.append(worst_fit)
        avg_fits.append(avg_fit)
        mut_rates.append(xy_sigma)

        # print stats
        print(f"====================\nGeneration {gen}\n====================")
        print(f"Best Fitness: {best_fit}")
        print(f"Worst Fitness: {worst_fit}")
        print(f"Avg. Fitness: {avg_fit}")

        if animate:
            # plot best & worst individual every <ANIMATE_CYCLE> generations
            if gen % ANIMATION_CYCLE == 0:
                best_ind = pop[fitness.index(max(fitness))]
                plot_solution(obstacles, furniture, best_ind, map_name, gen, "Best", open_reward, corner_reward, coverage_reward)

                worst_ind = pop[fitness.index(min(fitness))]
                plot_solution(obstacles, furniture, worst_ind, map_name, gen, "Worst", open_reward, corner_reward, coverage_reward)
        
        gen += 1
    
    # plot fitness trajectories
    plt.figure()
    plt.plot(avg_fits, label="Average Fitness")
    plt.plot(best_fits, label="Best Fitness")
    plt.plot(worst_fits, label="Worst Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    title = ""
    if map_name != None:
        title += f"Map: {map_name}\n"
    title += "Average Fitness and Best Fitness of Each Generation"
    plt.title(title)
    plt.show()

    return pop, fitness, gen

def visualize_solution(obstacles, furniture, individual):
    """
    Prints a visualization of an individual to the terminal.

    Parameters:
        obstacles (list):
            a 2D list where inner lists represent rows in the gridspace.
            Values of 0 are unoccupied cells, values of 1 are cells occupied by obstacles.
        furniture (list):
            a list of Furniture objects.
    """
    
    l = len(obstacles)
    w = len(obstacles[0])
    n = len(furniture)

    grid = copy.deepcopy(obstacles)

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

def plot_solution(obstacles, furniture, individual, map_name=None, gen=-1, rank=None, open_reward=True, corner_reward=True, coverage_reward=False):
    """
    Method for visualizing grid with placed furniture.
    Author: ChatGPT

    Parameters:
        obstacles (list): 
            a 2D list where inner lists represent rows in the gridspace.
            Values of 0 are unoccupied cells, values of 1 are cells occupied by obstacles.
        furniture (list): 
            a list of Furniture objects.
        individual (list):
            a potential solution represented by positions (Pos) for each Furniture object in the furniture parameter.
        map_name (str):
            the name of the map used, if chosen from the pre-made list. (DEFAULT = None)
        gen (int):
            the generation # of the current individual, for use in printing the plot title. Will read N/A if no generation # is input. (DEFAULT = -1)
        rank (str):
            the rank among the population of the individual being plotted. (DEFAULT = None)
        open_reward (Bool):
            whether or not to add reward for adjacent empty cells. (DEFAULT = True)
        corner_reward (Bool):
            whether or not to add reward for furniture-occupied corner cells. (DEFAULT = True)
        coverage_reward (Bool):
            whether or not to add reward for furniture-occupied cells. (DEFAULT = False)
    """

    l = len(obstacles)
    w = len(obstacles[0])

    fitness, gain, loss, oob, corners = eval_fit(obstacles, furniture, individual, open_reward, corner_reward, coverage_reward)

    BORDER = 4

    xmin = -BORDER
    xmax = w - 1 + BORDER
    ymin = -BORDER
    ymax = l - 1 + BORDER

    grid = [[[] for _ in range(w)] for _ in range(l)]

    # tracks furniture IDs in every visible cell (including OOB)
    visible_cells = {}

    furn_cells = []

    # obstacles
    for y in range(l):
        for x in range(w):
            if obstacles[y][x] == 1:
                grid[y][x].append(0)

    # project furniture
    for fid, furn in enumerate(furniture, start=1):

        pos = individual[fid-1]
        origin = pos.to_vec()

        cells = []

        for x in range(furn.w):
            for y in range(furn.l):

                if furn.occupancy[y][x] == 0:
                    continue

                rx, ry = furn.rotate(x, y, pos.rot)

                gx = origin.x + rx
                gy = origin.y + ry

                # interior grid tracking
                if 0 <= gx < w and 0 <= gy < l:
                    grid[gy][gx].append(fid)

                # visible area tracking
                if xmin <= gx <= xmax and ymin <= gy <= ymax:

                    cells.append((gx, gy))

                    key = (gx, gy)

                    if key not in visible_cells:
                        visible_cells[key] = [fid]
                    else:
                        visible_cells[key].append(fid)

        furn_cells.append((fid, cells))

    cell_size = 0.6

    fig_w = (xmax - xmin + 1) * cell_size
    fig_h = (ymax - ymin + 1) * cell_size
    fontsize = max(6, int(cell_size * 10))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.8)

    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(len(furniture))

    # shade entire extended area
    ax.add_patch(
        patches.Rectangle(
            (xmin, ymin),
            xmax-xmin+1,
            ymax-ymin+1,
            facecolor="black",
            alpha=0.1,
            zorder=0
        )
    )

    # draw real map background
    ax.add_patch(
        patches.Rectangle(
            (0,0),
            w,
            l,
            facecolor="white",
            zorder=1
        )
    )

    # gridlines
    for x in range(xmin, xmax+2):
        ax.plot([x, x], [ymin, ymax+1], color="lightgray", linewidth=0.7)

    for y in range(ymin, ymax+2):
        ax.plot([xmin, xmax+1], [y, y], color="lightgray", linewidth=0.7)

    # map boundary
    ax.plot([0, w], [0, 0], color="black", linewidth=2)
    ax.plot([0, w], [l, l], color="black", linewidth=2)
    ax.plot([0, 0], [0, l], color="black", linewidth=2)
    ax.plot([w, w], [0, l], color="black", linewidth=2)

    # obstacles
    for y in range(l):
        for x in range(w):

            if obstacles[y][x] == 1:

                rect = patches.Rectangle(
                    (x,y),
                    1,
                    1,
                    facecolor="black",
                    alpha=0.6,
                    zorder=4
                )

                ax.add_patch(rect)

    # draw furniture
    for fid, cells in furn_cells:

        color = cmap(fid-1)

        for (x,y) in cells:

            poly = patches.Polygon(
                [(x,y),(x+1,y),(x+1,y+1),(x,y+1)],
                facecolor=color,
                edgecolor="black",
                alpha=0.6,
                zorder=3
            )

            ax.add_patch(poly)

    # labels for ALL visible cells
    for (x,y), ids in visible_cells.items():

        label = ",".join(str(i) for i in ids)

        text_color = "black"

        if 0 <= x < w and 0 <= y < l:
            if 0 in grid[y][x]:
                text_color = "white"

        ax.text(
            x + 0.5,
            y + 0.5,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            zorder=5
        )

    ax.set_xlim(xmin, xmax+1)
    ax.set_ylim(ymax+1, ymin)
    ax.set_aspect("equal")

    title = ""

    if map_name is not None:
        title += f"Map: {map_name}\n"

    title += "Furniture Layout"

    if rank is not None:
        title += " | " + rank + " Individual "

    if gen < 0:
        gen = "N/A"

    title += f"| Generation: {gen}\nFitness: {fitness:.2f} | Gain: {gain:.2f} | Loss: {loss:.2f}"

    title += f"\nOOB Count: {oob}"

    if corner_reward:
        title += f" | Corner Count: {corners}"

    if not open_reward or not corner_reward:
        title += "\n"

    if not open_reward:
        title += "(Open Reward OFF)"

    if not corner_reward:
        title += " | (Corner Reward OFF)"

    ax.set_title(title)

    plt.show()

def eval_fit(obstacles, furniture, individual, open_reward=True, corner_reward=True, coverage_reward=False):
    """
    Calculates fitness of individuals in the population.
    Higher value => better fitness.

    Parameters:
        obstacles (list):
            a 2D list where inner lists represent rows in the gridspace.
            Values of 0 are unoccupied cells, values of 1 are cells occupied by obstacles.
        furniture (list): 
            a list of Furniture objects.
        individual (list):
            a potential solution represented by positions (Pos) for each Furniture object in the furniture parameter.
        open_reward (Bool):
            whether or not to add reward for adjacent empty cells. (DEFAULT = True)
        corner_reward (Bool):
            whether or not to add reward for furniture-occupied corner cells. (DEFAULT = True)
        coverage_reward (Bool):
            whether or not to add reward for furniture-occupied cells. (DEFAULT = False)

    Returns:
        reward (float): 
            computed sum of loss (out-of-bounds, overlap) and gain (adjacent empty cells).
        gain (float):
            OPEN_FACTOR * (# of unique pairs of adjacent empty cells).
        loss (float):
            (number of overlapping furniture/obstacles cells) + (OOB_PENALTY * (out-of-bounds furniture cells)).
        total_oob (int):
            the number of furniture-occupied out-of-bounds cells.
        corners_occupied (int):
            the number of corners occupied by furniture. Corners are only considered when not blocked by obstacles.
    """
    l = len(obstacles)
    w = len(obstacles[0])
    n = len(furniture)

    OOB_PENALTY = 3
    OPEN_FACTOR = 0.05 # should be much lower than 1
    COVERAGE_REWARD = 3

    occupancy = obstacles.copy()
    loss = 0.0
    gain = 0.0

    #Bonus for corners
    corners = {(0,0),(w-1,0), (0, l-1), (w-1,l-1)}
    corner_bonus_count = 4
    corner_bonus = 2


    # calculate out-of-bounds penalty
    for i in range(n):
        pos = individual[i]
        loss -= furniture[i].project_to(obstacles, occupancy, pos)
    total_oob = int(-loss)
    loss *= OOB_PENALTY

    # calculate overlap
    overlap = np.maximum(occupancy - 1, 0)
    loss -= overlap.sum()

    empty = (occupancy == 0)

    if open_reward:
        # calculate openness

        #Calculate Open Space Reward with both vertical and horizontal adjacency 
        gain += (
            np.sum(empty[:,1 :] & empty[:, :-1]) + #horizontal 
            np.sum(empty[1:, :] & empty[:-1, :]) #vertical
        )

        gain *= OPEN_FACTOR
    
    corners_occupied = 0
    if corner_reward:
        #add bonus for being in the corner when corner is not blocked by obstacle
        for(cx, cy) in corners:
            if occupancy[cy][cx] > obstacles[cy][cx]:
                gain += corner_bonus
                corners_occupied += 1
    
    if coverage_reward:
        gain += COVERAGE_REWARD * np.count_nonzero(occupancy)

    reward = loss + gain

    return reward, gain, loss, total_oob, corners_occupied

def mutate(individual, swap_chance=100, xy_sigma=0.5, rot_sigma=0.5):
    """
    Mutates an individual.

    Parameters:
        individual (list):
            an individual of the population represented by positions (Pos) for each Furniture object.
        swap_chance (int):
            each position in the individual will have a 1 in swap_chance chance of swapping places with another position in the individual. (DEFAULT = 100)
        xy_sigma (float):
            the standard deviation of the gaussian from which the mutated x,y coordinated will be sampled. (DEFAULT = 0.5)
        rot_sigma (float):
            the standard deviation of the gaussian from which the mutated rotation will be sampled. (DEFAULT = 0.5)
    """

    #mut_individual = copy.deepcopy(individual)
    n = len(individual)

    # swap mutations
    for p in range(n):
        swap_chance = max(swap_chance,1)
        if random.randint(0,swap_chance-1) == 0:
            if n > 1:
                # get random index other than p. Do this by sampling uniformly from 0 to max_index - 1 (i.e. 0 to n-2)
                sp = random.randint(0,n-2)

                # ensure all indices can be covered and p is excluded
                if sp >= p:
                    sp += 1

                # perform swap
                individual[p], individual[sp] = individual[sp], individual[p]

    # point mutations
    for pos in individual:
        # sample mutated values using gaussian
        mut_x = random.gauss(mu=pos.x, sigma=xy_sigma)
        mut_y = random.gauss(mu=pos.y, sigma=xy_sigma)
        mut_rot = random.gauss(mu=pos.rot.value, sigma=rot_sigma)

        # convert sampled values to integers, with rounding bias towards mu (i.e. pre-mutation values)
        mut_x = math.floor(mut_x) if mut_x >= pos.x else math.ceil(mut_x)
        mut_y = math.floor(mut_y) if mut_y >= pos.y else math.ceil(mut_y)
        mut_rot = math.floor(mut_rot) if mut_rot >= pos.rot.value else math.ceil(mut_rot)

        # ensure rotation value is  valid
        mut_rot = mut_rot % 4

        # set mutated values
        pos.set_x(mut_x)
        pos.set_y(mut_y)
        pos.set_rot(mut_rot)

def crossover(parent1, parent1_fit_window, parent2, parent2_fit_window, reprod_fit=False):
    """
    Produces child from 2 parents. Can employ fitness-proportionate inheritance using fitness windows to mitigate scaling effects.

    Paremeters:
        parent1 (list):
            an individual of the population represented by positions (Pos) for each Furniture object.
        parent1_fit_window (int):
            fitness window of parent1.
        parent2 (list):
            an individual of the population represented by positions (Pos) for each Furniture object.
        parent2_fit_window (int):
            fitness window of parent2.
        reprod_fit (Bool):
            reproductive fitness. whether the offspring's chance to inherit a trait from each parent is proportional that parent's fitness window.
            If False, chance will be 50/50 (DEFAULT=False)
            
    Returns:
        child (list):
            an individual produced by crossover of 2 parent individuals, represented by positions (Pos) for each Furniture object.
    """

    n = len(parent1)
    child = copy.deepcopy(parent1)

    for i in range(n):
        # get the Pos at index i of parent2 as an iterable
        p2_pos = parent2[i].to_tuple()

        for j in range(len(p2_pos)):
            if reprod_fit:
                use_p2 = False
                # determine chance to use parent2's trait
                if parent2_fit_window > 0:
                    p2_ratio = parent2_fit_window/(parent1_fit_window + parent2_fit_window)
                    if random.random() < p2_ratio:
                        use_p2 = True
            else:
                # 50/50 to use each attribute of parent2
                use_p2 = random.choice([True,False])

            if use_p2:
                # get the attribute of parent2's to use (i.e. x, y or rotation)
                att = p2_pos[j]
                if j == 0:
                    # update x
                    child[i].set_x(att)
                elif j == 1:
                    # update y
                    child[i].set_y(att)
                elif j == 2:
                    # update rot
                    child[i].set_rot(att)

    return child

def tournament(pop, fitness, mating_pool_size, tournament_size):
    """
    Tournament selection without replacement, higher fitness = better.
    Adapted from Aidan's Coding Exercise 1.

    Paremeters:
        pop (list):
            a 2D list where inner lists are solutions represented by positions (Pos) for each Furniture object in the furniture parameter.
            Represents the current population.
        fitness (list):
            a list representing the fitness values of each individual in the population.
        mating_pool_size (int):
            the number of individuals to be selected to mate.
        tournament_size (int):
            the number of individuals to participate in each tournament (higher => more selection pressure).

    Returns:
        selected_to_mate (list):
            a list of tuples representing individuals selected for mating and their fitnesses.
    """

    selected_to_mate = []
    n = len(fitness)

    while len(selected_to_mate)//2 < mating_pool_size:
        # generate random indices to participate in tournament
        tournament_indices = random.sample(range(n),tournament_size)
        
        # initialize 1st and 2nd place
        first = tournament_indices[0]
        second = tournament_indices[1]

        # compare fitnesses
        for i in range(0,len(tournament_indices)):
            # if new individual has highest fitness so far
            if fitness[tournament_indices[i]] >= fitness[first]:
                # update both frontrunners
                second = first
                first = tournament_indices[i]

            # if new individual has second highest fitness so far
            elif fitness[tournament_indices[i]] >= fitness[second]:
                # update 2nd place
                second = tournament_indices[i]

        selected_to_mate.append((pop[first], pop[second], fitness[first], fitness[second]))

    return selected_to_mate

def sort_population(pop, fitness):
    """
    Sorts a population by fitness (descending value).
    Borrowed from Coding Exercise 1.

    Parameters:
        pop (list):
            a 2D list where inner lists are solutions represented by positions (Pos) for each Furniture object in the furniture parameter.
            Represents the current population.
        fitness (list):
            a list representing the fitness values of each individual in the population.

    Returns:
        sorted_pop (list):
            a sorted copy of pop (descending fitness).
        sorted_fitness (list):
            a sorted copy of fitness (descending).
    """
    pop_fit_gain_loss_tup = list(map(list, zip(pop, fitness)))
    pop_fit_gain_loss_tup.sort(key=operator.itemgetter(1), reverse=True)
    sorted_pop = []
    sorted_fit = []
    for entry in pop_fit_gain_loss_tup:
        sorted_pop.append(entry[0])
        sorted_fit.append(entry[1])
    return sorted_pop, sorted_fit

def replacement(pop, fitness, offspring, offspring_fitness):
    """
    Offspring to replace the worst individuals in the current generation.

    Parameters:
        pop (list):
            a 2D list where inner lists are solutions represented by positions (Pos) for each Furniture object in the furniture parameter.
            Represents the current population.
        fitness (list):
            a list representing the fitness values of each individual in the population.
        offspring (list):
            a 2D list where inner lists are solutions represented by positions (Pos) for each Furniture object in the furniture parameter.
            Represents the offspring that will replace members of the population.
        offspring_fitness (list):
            a list representing the fitness values of each offspring.
        
    Returns:
        new_pop (list):
            the population with the n worst members replaced by the n offspring.
        new_fitness (list):
            a list representing the fitness values of each individual in the new population.
    """

    new_pop = []
    new_fitness = []
    new_pop, new_fitness = sort_population(copy.deepcopy(pop), copy.deepcopy(fitness))
    k = len(pop) - len(offspring)

    # replace worst members of population with offspring
    new_pop[k:] = offspring
    new_fitness[k:] = offspring_fitness

    #random.shuffle(new_pop)
    #random.shuffle(new_fitness)

    return new_pop, new_fitness

class Furniture:
    """
    Holds the furniture shape.

    Parameters:
        occupancy (list): 
            a 2D list where inner lists represent the rows of the rectangle enclosing the furniture. 
            Values > 0 in the rows indicate how many furniture items occupy that cell; values of 0 are cells that are unoccupied.

    Attributes:
        occupancy (list): 
            a 2D list where inner lists represent the rows of the rectangle enclosing the furniture. 
            Values of 1 in the rows indicate cells occupied by the furniture; values of 0 are cells that are unoccupied.
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
                the rotated (x,y) coordinate.
        """

        # rotation transformations
        ROT_MATRIX = {
            Pos.DIR.CW0: lambda x,y: (  x  ,   y  ),
            Pos.DIR.CW90:  lambda x,y: (self.l-1-y,   x  ),
            Pos.DIR.CW180: lambda x,y: (self.w-1-x, self.l-1-y),
            Pos.DIR.CW270:  lambda x,y: (  y  , self.w-1-x)   
        }

        return ROT_MATRIX[rot](x,y)

    def project_to(self, obstacles, grid_occupancy, pos):
        """
        Projects the furniture object onto a grid, updating the occupancy and overlap values.

        Parameters:
            obstacles (list):
                a 2D list where inner lists represent rows in the gridspace.
                Values of 0 are unoccupied cells, values of 1 are cells occupied by obstacles.
            grid_occupancy (list):
                same shape as obstacles, where values in the grid represent the number of furniture objects occupying the cell.
            overlap (list):
                same shape as obstacles, where values in the grid represent the number of overlaps of furniture and obstacles in the cell.
            pos (Pos):
                the position to project the furniture object to.

        Returns:
            oob (int):
                the number of cells occupied by the furniture that fall out-of-bounds.
        """

        l = len(grid_occupancy)
        w = len(grid_occupancy[0])

        origin = pos.to_vec()

        oob = 0

        for x in range(self.w):
            for y in range(self.l):

                rx, ry = self.rotate(x, y, pos.rot)

                offset = Vec(rx, ry)
                offset.add(origin)

                if (not(0 <= offset.x <= w-1)) or (not(0 <= offset.y <= l-1)) or (obstacles[offset.y][offset.x] == 1):
                    # if the (x,y) coordinate is out-of-bounds or an obstacle space, increment oob
                    oob += self.occupancy[y][x] 
                else:
                    # update occupancy for in-bounds (x,y) coordinate
                    grid_occupancy[offset.y][offset.x] += self.occupancy[y][x]

        return oob

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
        DIR.CW270 : "270° Clockwise",
        DIR.CCW0 : "0° Counter Clockwise",
        DIR.CCW90 : "90° Counter Clockwise",
        DIR.CCW180 : "180° Counter Clockwise",
        DIR.CCW270 : "270° Counter Clockwise",
        DIR.NORTH : "NORTH",
        DIR.EAST : "EAST",
        DIR.SOUTH : "SOUTH",
        DIR.WEST : "WEST"
    }

    def __init__(self, x=0, y=0, rot=0):
        self.x = x
        self.y = y
        self.rot = rot

    def __str__(self):
        return f"({self.x}, {self.y}, {self.rot})"

    def to_vec(self):
        """
        Converts x, y coordinates to Vec object.
        """
        return Vec(self.x, self.y)
    
    def to_tuple(self):
        """
        Converts Pos object to tuple of form (self.x, self.y, self.rot).
        """
        return self.x, self.y, self.rot
    
    def set_x(self, x):
        """
        Sets the x coordinate.
        """
        self.x =x

    def set_y(self, y):
        """
        Sets the y coordinate.
        """
        self.y = y
    
    def set_rot(self, rot):
        """
        Sets the rotation.
        """
        if isinstance(rot,int):
            rot = Pos.DIR(rot)
        self.rot = rot

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

class BoundingBox:
    """
    Defines grid boundaries. Currently not used.

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

# some custom maps
# note that with the furniture defined in furnitures below, it is impossible to have no overlap and no oob in Big+
maps = {
    "Empty10x10": np.array(
        [[0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0]]),
                        
    "Split": np.array(
        [[0,0,0,0,0,1,1,0,0,0,0,0],
         [0,0,0,0,0,1,1,0,0,0,0,0],
         [0,0,0,0,0,1,1,0,0,0,0,0],
         [0,0,0,0,0,1,1,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,1,1,0,0,0,0,0],
         [0,0,0,0,0,1,1,0,0,0,0,0],
         [0,0,0,0,0,1,1,0,0,0,0,0],
         [0,0,0,0,0,1,1,0,0,0,0,0]]),
                        
    "Hall": np.array(
        [[0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0]]),
                        
    "NarrowHall": np.array(
        [[0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [1,0,0,0,0,1],
         [1,0,0,0,0,1],
         [1,0,0,0,0,1],
         [1,0,0,0,0,1],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0]]),

    "BigL": np.array(
        [[1,1,1,1,1,0,0,0,0,0],
         [1,1,1,1,1,0,0,0,0,0],
         [1,1,1,1,1,0,0,0,0,0],
         [1,1,1,1,1,0,0,0,0,0],
         [1,1,1,1,1,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0]]),

    "Compact": np.array(
        [[1,1,1,1,1,0,0,0,0,0],
         [1,1,1,1,1,0,0,0,0,0],
         [1,1,1,1,1,0,0,1,0,0],
         [1,1,1,1,1,0,0,0,0,0],
         [1,1,1,1,1,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,1,1,1,0,0,0,1,0,0],
         [0,1,1,1,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,1,1]]),

    "Big+":np.array(
        [[1,1,1,0,0,0,0,1,1,1],
         [1,1,1,0,0,0,0,1,1,1],
         [1,1,1,0,0,0,0,1,1,1],
         [1,1,1,0,0,0,0,1,1,1],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [1,1,1,0,0,0,0,1,1,1],
         [1,1,1,0,0,0,0,1,1,1],
         [1,1,1,0,0,0,0,1,1,1],
         [1,1,1,0,0,0,0,1,1,1]]),
                        
    "Small": np.array(
        [[0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0],
         [0,0,0,0,0,0]]),
         
    "BigX": np.array(
        [[0,0,0,1,1,1,1,1,0,0,0],
         [0,0,0,0,1,1,1,0,0,0,0],
         [0,0,0,0,0,1,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0],
         [1,0,0,0,0,0,0,0,0,0,1],
         [1,1,0,0,0,0,0,0,0,1,1],
         [1,0,0,0,0,0,0,0,0,0,1],
         [0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,1,0,0,0,0,0],
         [0,0,0,0,1,1,1,0,0,0,0],
         [0,0,0,1,1,1,1,1,0,0,0]])}

# 6 different furniture items
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
               [1,0]]),
    Furniture([[1,1,1,1,1],
               [1,1,1,1,1],
               [1,1,0,0,1],
               [1,1,0,0,0],
               [1,1,0,0,0]]),
    Furniture([[1,1,1,1,1,1,1]])
]

# tiling puzzle (goal: fill all spaces in layout, no overlap - our EA is not necessarily designed for this but can be applied!)
tiling_maps = {"5x5": np.array(
        [[0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0],
         [0,0,0,0,0]]),
               "Pentomino": np.array(
        [[0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0]]
               )}

pieces_5x5 = [
    Furniture([[1,1],
               [1,0]]),
    Furniture([[1,1],
               [1,1]]),
    Furniture([[1,1,1],
               [0,1,1]]),
    Furniture([[1,0,0,0],
               [1,1,1,1]]),
    Furniture([[1,1],
               [0,1],
               [0,1]]),
    Furniture([[1,1,1,1]])
]

pieces_pentomino = [
    Furniture([[0,1,0],
               [1,1,1],
               [0,0,1]]),
    Furniture([[1,1,1,1,1]]),
    Furniture([[1,1,1,1],
               [1,0,0,0]]),
    Furniture([[1,1,0,0,0],
               [0,1,1,1,1]]),
    Furniture([[1,1,1],
               [0,1,1]]),
    Furniture([[1,1,1],
               [0,1,0],
               [0,1,0]]),
    Furniture([[1,0,1],
               [1,1,1]]),
    Furniture([[1,0,0],
               [1,0,0],
               [1,1,1]]),
    Furniture([[1,0,0],
               [1,1,0],
               [0,1,1]]),
    Furniture([[0,1,0],
               [1,1,1],
               [0,1,0]]),
    Furniture([[0,0,1,0],
               [1,1,1,1]]),
    Furniture([[1,1,0],
               [0,1,0],
               [0,1,1]])
]

def run_all_maps(maps, furniture, fit_thresh=np.inf, animate=False, open_reward=True, corner_reward=True, coverage_reward=False, reprod_fit=False, const_mut=True, immigration=True):
    """
    Runs all layouts in a dictionary through the EA.

    Parameters:
        maps (dict):
            a dictionary of name (str) : layout (np.array) pair representing the maps to run through the EA.
        furniture (list): 
            a list of Furniture objects to be placed in each layout.
        fit_thresh (float):
            a fitness threshold to stop generations when the best member has fitness >= this value. 
            Default is infinity, meaning runs will always go until the GENERATION_COUNT. (DEFAULT = np.inf)
        animate (Bool):
            if True, plot the best and worst individual from certain generations as evolution progresses. (DEFAULT = False)
        open_reward (Bool):
            whether or not to add reward for adjacent empty cells. (DEFAULT = True)
        corner_reward (Bool):
            whether or not to add reward for furniture-occupied corner cells. (DEFAULT = True)
        coverage_reward (Bool):
            whether or not to add reward for furniture-occupied cells. (DEFAULT = False)
        reprod_fit (Bool):
            reproductive fitness. whether the offspring's chance to inherit a trait from each parent is proportional that parent's fitness.
            If False, chance will be 50/50 (DEFAULT=False)
        const_mut (Bool):
            whether mutation rates remain constant throughout evolution. Rates will decrease with generations if set to False. (DEFAULT = True)
        immigration (Bool):
            whether randomly generated individuals should be added (with replacement) to the population at set intervals. (DEFAULT = True)
    """
    for name, layout in maps.items():
        pop, fit, gen = fit_furniture(layout, furniture, name, fit_thresh, animate, open_reward, corner_reward, coverage_reward, reprod_fit, const_mut, immigration)
        best_fit = max(fit)
        worst_fit = min(fit)
        best_ind = pop[fit.index(best_fit)]
        worst_ind = pop[fit.index(worst_fit)]
        plot_solution(layout, furniture, best_ind, name, gen, "Best", open_reward, corner_reward, coverage_reward)
        plot_solution(layout, furniture, worst_ind, name, gen, "Worst", open_reward, corner_reward, coverage_reward)

def run_map(maps, furniture, name, fit_thresh=np.inf, animate=False, open_reward=True, corner_reward=True, coverage_reward=False, reprod_fit=False, const_mut=True, immigration=True):
    """
    Runs a specified layout from the maps directory.
    
    Parameters:
        maps (dict):
            a dictionary of name (str) : layout (np.array) pair representing the maps to run through the EA.
        furniture (list): 
            a list of Furniture objects to be placed in each layout.
        name (str):
            the name key of the desired layout.
        fit_thresh (float):
            a fitness threshold to stop generations when the best member has fitness >= this value. 
            Default is infinity, meaning runs will always go until the GENERATION_COUNT. (DEFAULT = np.inf)
        animate (Bool):
            if True, plot the best and worst individual from certain generations as evolution progresses. (DEFAULT = False)
        open_reward (Bool):
            whether or not to add reward for adjacent empty cells. (DEFAULT = True)
        corner_reward (Bool):
            whether or not to add reward for furniture-occupied corner cells. (DEFAULT = True)
        coverage_reward (Bool):
            whether or not to add reward for furniture-occupied cells. (DEFAULT = False)
        reprod_fit (Bool):
            reproductive fitness. whether the offspring's chance to inherit a trait from each parent is proportional that parent's fitness.
            If False, chance will be 50/50 (DEFAULT=False)
        const_mut (Bool):
            whether mutation rates remain constant throughout evolution. Rates will decrease with generations if set to False. (DEFAULT = True)
        immigration (Bool):
            whether randomly generated individuals should be added (with replacement) to the population at set intervals. (DEFAULT = True)
    """
    layout = maps[name]
    pop, fit, gen = fit_furniture(layout, furniture, name, fit_thresh, animate, open_reward, corner_reward, coverage_reward, reprod_fit, const_mut, immigration)
    best_fit = max(fit)
    worst_fit = min(fit)
    best_ind = pop[fit.index(best_fit)]
    worst_ind = pop[fit.index(worst_fit)]
    plot_solution(layout, furniture, best_ind, name, gen, "Best", open_reward, corner_reward, coverage_reward)
    plot_solution(layout, furniture, worst_ind, name, gen, "Worst", open_reward, corner_reward, coverage_reward)
 
##########
# OUTPUT #
##########
#run_map(tiling_maps, pieces_pentomino, "Pentomino", fit_thresh=180.0, animate=False, open_reward=False, corner_reward=False, coverage_reward=True, reprod_fit=True, const_mut=True, immigration=True)
#run_map(maps, furnitures, "BigX", animate=False, open_reward=True, corner_reward=True, coverage_reward=False, reprod_fit=False, const_mut=False, immigration=True)
#run_all_maps(maps, furnitures, animate=False, open_reward=True, corner_reward=True, coverage_reward=False, reprod_fit=False, const_mut=False, immigration=True)