"""
ROB521_assignment1.py

This assignment will introduce you to the idea of motion planning for  
holonomic robots that can move in any direction and change direction of 
motion instantaneously.  Although unrealistic, it can work quite well for
complex large scale planning.  You will generate mazes to plan through 
and employ the PRM algorithm presented in lecture as well as any 
variations you can invent in the later sections.

There are three questions to complete (5 marks each):

    Question 1: implement the PRM algorithm to construct a graph
    connecting start to finish nodes.
    Question 2: find the shortest path over the graph by implementing the
    Dijkstra's or A* algorithm.
    Question 3: identify sampling, connection or collision checking 
    strategies that can reduce runtime for mazes.

Three helper functions are provided for you to use in your motion planning 
solution: min_dist_to_edges, distance_point_to_segment, and check_collision.
The first two are used to determine if a point is at least a minimum distance
from all walls in the maze, and the third checks if a line segment intersects
any walls in the maze.  You may modify these functions if you wish or use them
as is.
 
To complete the assignment, fill in the required sections of this script with 
your code, run it to generate the requested plots, then paste the plots into 
a short report that includes a few comments about what you've observed.  
Append your version of this script to the report.  Hand in the report as a 
PDF file.

requires: numpy, matplotlib

S L Waslander, revised January 2026 - Converted to Python...
"""

import numpy as np
import matplotlib.pyplot as plt
from time import time
from math import sqrt, inf

# set random seed for repeatability if desired
# np.random.seed(1)

# ==========================
# Maze Generation
# ==========================
#
# The maze function returns a map object with all of the edges in the maze.
# Each row of the map structure draws a single line of the maze.  The
# function returns the lines with coordinates [x1 y1 x2 y2].
# Bottom left corner of maze is [0.5 0.5], 
# Top right corner is [col+0.5 row+0.5]
# Each wall is [start_col start_row end_col end_row] and goes from bottom/left to top/right.

def maze(rows, cols):
    """
    Generate a random maze using recursive depth-first search algorithm.
    Returns a list of line segments representing walls.
    """
    # Initialize grid with all walls
    walls = []
    
    # Create walls list, outer walls first
    for i in range(cols):
        walls.append([i + 0.5, 0.5, i + 1.5, 0.5])  
        for j in range(rows):
            if i == 0:
                walls.append([0.5, j + 0.5, 0.5, j + 1.5])  
            walls.append([i + 0.5, j + 1.5, i + 1.5, j + 1.5])  # horizontal walls
            walls.append([i + 1.5, j + 0.5, i + 1.5,  j + 1.5])  # vertical walls
  
    visited = np.zeros((cols, rows), dtype=bool)
    
    #Remove start and end walls
    walls.remove([0.5, 0.5, 0.5, 1.5])  # Remove entrance wall
    walls.remove([cols + 0.5, rows - 0.5, cols + 0.5, rows + 0.5])  # Remove exit wall

    def carve_path(x, y, walls):
        visited[x, y] = True
        # Directions: right, down, left, up
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        np.random.shuffle(directions)
        
        for (dx, dy) in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and not visited[nx, ny]:
                # Remove wall between current and next cell
                if dx == 1:  # right
                    wall_to_remove = [x + 1.5, y + 0.5, x + 1.5, y + 1.5]
                    if wall_to_remove in walls:
                        walls.remove(wall_to_remove)
                elif dx == -1:  # left
                    wall_to_remove = [x + 0.5, y + 0.5, x + 0.5, y + 1.5]
                    if wall_to_remove in walls:
                        walls.remove(wall_to_remove)
                elif dy == 1:  # up
                    wall_to_remove = [x + 0.5, y + 1.5, x + 1.5, y + 1.5]
                    if wall_to_remove in walls:
                        walls.remove(wall_to_remove)
                elif dy == -1:  # down
                    wall_to_remove = [x + 0.5, y + 0.5, x + 1.5, y + 0.5]
                    if wall_to_remove in walls:
                        walls.remove(wall_to_remove)
                carve_path(nx, ny, walls)

    carve_path(0, 0, walls)
    return walls


def show_maze(walls, rows, cols, ax):
    """Draw the maze on the given matplotlib axis."""
    walls = np.array(walls)
    for wall in walls:
        ax.plot([wall[0], wall[2]], [wall[1], wall[3]], 'k-', linewidth=1)
    ax.set_xlim(0, cols + 1)
    ax.set_ylim(0, rows + 1)
    ax.set_aspect('equal')
    ax.grid(False)


def min_dist_to_edges(point, walls, min_dist=0.1):
    """
    Check if a point is at least min_dist away from all walls.
    Returns True if the point is valid (far enough from all walls).
    """
    px, py = point
    for wall in walls:
        x1, y1, x2, y2 = wall
        # Distance from point to line segment
        dist = distance_point_to_segment(px, py, x1, y1, x2, y2)
        if dist < min_dist:
            return False
    return True


def distance_point_to_segment(px, py, x1, y1, x2, y2):
    """Calculate minimum distance from point (px, py) to line segment."""
    # Vector from start to end of segment
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return sqrt((px - x1)**2 + (py - y1)**2)
    
    # Parameter t for closest point on line
    t = max(0, min(1, ((px - x1)*dx + (py - y1)*dy) / length_sq))
    
    closest_x = x1 + t*dx
    closest_y = y1 + t*dy
    
    return sqrt((px - closest_x)**2 + (py - closest_y)**2)


def check_collision(x1, y1, x2, y2, walls, min_dist=0.1):
    """
    Check if the line segment from (x1,y1) to (x2,y2) collides with any walls.
    Returns True if collision-free.
    """
    # Check multiple points along the path
    steps = int(sqrt((x2-x1)**2 + (y2-y1)**2) * 10) + 2
    for i in range(steps):
        t = i / (steps - 1) if steps > 1 else 0
        x = x1 + t*(x2 - x1)
        y = y1 + t*(y2 - y1)
        if not min_dist_to_edges([x, y], walls, min_dist):
            return False
    return True


# ======================================================
# Question 1: construct a PRM connecting start and finish
# ======================================================
#
# Using 500 samples, construct a PRM graph whose milestones stay at least 
# 0.1 units away from all walls, using the min_dist_to_edges function provided for 
# collision detection.  Use a nearest neighbour connection strategy and the 
# check_collision function provided for collision checking, and find an 
# appropriate number of connections to ensure a connection from  start to 
# finish with high probability.

row = 5
col = 7
walls = maze(row, col)
start = np.array([0.5, 1.0])
finish = np.array([col + 0.5, row])

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(start[0], start[1], 'go', markersize=8)
ax.plot(finish[0], finish[1], 'rx', markersize=8)
show_maze(walls, row, col, ax)
plt.draw()
plt.pause(0.1)

# variables to store PRM components
nS = 500  # number of samples to try for milestone creation
milestones = [start, finish]  # each row is a point [x y] in feasible space
edges = []  # each row an edge of the form [x1 y1 x2 y2]

print("Time to create PRM graph")
t0 = time()

# ------insert your PRM generation code here-------
# The PRM algorithm is as follows:
# 1. Sample a random point in the maze
# 2. Check if the point is valid (at least 0.1 units away from walls, using min_dist_to_edges
# 3. If valid, add to milestones
# 4. For each new milestone, find nearest neighbors and attempt to connect them using check_collision (the number of neighbours it connects to is tunable)

# Define the number of nearest neighbors to attempt connections with
k_neighbors = 5 # can tune this depending on the performance desired

# Sampling a random 500 points in the maze and convert to maze coorindates since it's offset by 0.5
while len(milestones) < nS:
    x_rand = np.random.uniform(0.5, col + 0.5)
    y_rand = np.random.uniform(0.5, row + 0.5)

    # These will just be candidate samples before we check for the distance to objects
    candidate = np.array([x_rand, y_rand])
    
    # Check if the candidate is valid by using min_dist_to_edges to ensure it's at least 0.1 units away from walls
    if min_dist_to_edges(candidate, walls):
        milestones.append(candidate)

# Now that sampling is done, we want to connect the milestones to the k-nearest neighbors
milestones = np.array(milestones)    

for idx, point in enumerate(milestones): # idx is the index in the list of milstones and point is the actual coordinate (will loop over this)
    # Before we connect to its nearest neighbors, we need to calculate the distances to all other points to find who the nearest neighbors are
    # For each iteration, we take one point and compute its Euclidean distance to all other points, then use it to find the nearest neighbors and then discard and move to the next point
    diffs = milestones - point 
    euclidean_dists = np.sqrt(np.sum(diffs**2, axis=1))
    nearest_indices = np.argsort(euclidean_dists)  # Sort indices based on distance
    
    # Now, we only select the k-nearest neighbors (skipping the first one since it's the point itself)
    neighbors = nearest_indices[1:k_neighbors + 1]
    
    # Attempt to connect to each of the k-nearest neighbors
    for neighbor_idx in neighbors:
        neighbor_point = milestones[neighbor_idx]
        # Check if the edge between point and neighbor_point is collision-free, if so, add to edges
        if check_collision(point[0], point[1], neighbor_point[0], neighbor_point[1], walls):
            edges.append([point[0], point[1], neighbor_point[0], neighbor_point[1]])
     
# ------end of your PRM generation code -------
elapsed = time() - t0
print(f"Time elapsed: {elapsed:.4f} seconds")

ax.plot(np.array(milestones)[:, 0], np.array(milestones)[:, 1], 'm.', markersize=4)
if edges:
    edges = np.array(edges)
    ax.plot([edges[:, 0], edges[:, 2]], [edges[:, 1], edges[:, 3]], 'magenta', alpha=0.5, linewidth=0.5)
ax.set_title(f'Q1 - {row} X {col} Maze PRM')
plt.tight_layout()
plt.savefig('assignment1_q1.png', dpi=150)
plt.show()


# =================================================================
# Question 2: Find the shortest path over the PRM graph
# =================================================================
#
# Using an optimal graph search method (Dijkstra's or A*), find the 
# shortest path across the graph generated.  Please code your own 
# implementation instead of using any built in functions.

print('Time to find shortest path')
t0 = time()
spath = []  # list of milestone indices that form the shortest path

# ------insert your shortest path finding code here-------











# ------end of your shortest path finding code -------


elapsed = time() - t0
print(f"Time elapsed: {elapsed:.4f} seconds")


# plot the shortest path
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(np.array(milestones)[:, 0], np.array(milestones)[:, 1], 'm.', markersize=4)
if edges:
    edges = np.array(edges)
    ax.plot([edges[:, 0], edges[:, 2]], [edges[:, 1], edges[:, 3]], 'magenta', alpha=0.5, linewidth=0.5)
ax.plot(start[0], start[1], 'go', markersize=8)
ax.plot(finish[0], finish[1], 'rx', markersize=8)

if len(spath) > 1:
    path_points = milestones[spath]
    ax.plot(path_points[:, 0], path_points[:, 1], 'go-', linewidth=3, markersize=6)

show_maze(walls, row, col, ax)
ax.set_title(f'Q2 - {row} X {col} Maze Shortest Path')
plt.tight_layout()
plt.savefig('assignment1_q2.png', dpi=150)
plt.show()


# ================================================================
# Question 3: find a faster way
# ================================================================
#
# Modify your milestone generation, edge connection, collision detection 
# and/or shortest path methods to reduce runtime.  What is the largest maze 
# for which you can find a shortest path from start to goal in under 20 
# seconds on your computer? (Anything larger than 40x40 will suffice for 
# full marks)

row = 25
col = 25
walls = maze(row, col)
start = np.array([0.5, 1.0])
finish = np.array([col + 0.5, row])
milestones = [list(start), list(finish)]
edges = []

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(start[0], start[1], 'go', markersize=8)
ax.plot(finish[0], finish[1], 'rx', markersize=8)
show_maze(walls, row, col, ax)
plt.draw()
plt.pause(0.1)

print(f"Attempting large {row} X {col} maze...")
t0 = time()

# ------insert your optimized algorithm here------











# ------end of your optimized algorithm-------
dt = time() - t0

ax.plot(np.array(milestones)[:, 0], np.array(milestones)[:, 1], 'm.', markersize=2)
if edges:
    edges = np.array(edges)
    ax.plot([edges[:, 0], edges[:, 2]], [edges[:, 1], edges[:, 3]], 'magenta', alpha=0.3, linewidth=0.3)

if len(spath) > 1:
    path_points = milestones[spath]
    ax.plot(path_points[:, 0], path_points[:, 1], 'go-', linewidth=2, markersize=4)

ax.set_title(f'Q3 - {row} X {col} Maze solved in {dt:.4f} seconds')
plt.tight_layout()
plt.savefig('assignment1_q3.png', dpi=150)
plt.show()

print(f"Q3 completed in {dt:.4f} seconds")
