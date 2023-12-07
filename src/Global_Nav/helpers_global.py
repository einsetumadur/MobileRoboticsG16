import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import cv2

### UTILE 


# give the path with point of changing direction
def global_final(occupancy_grid, start, goal, movement , plot=False): 
    occupancy_grid = cv2.flip(occupancy_grid,1)
    
    if plot:
        print('Initial Map:')
        fig, ax =create_empty_plot(occupancy_grid.shape[0], occupancy_grid.shape[1])
        cmap = colors.ListedColormap(['white', 'black']) # Select the colors with which to display obstacles and free cells
        ax.imshow(occupancy_grid.transpose(), cmap=cmap)
        ax.scatter(start[0], start[1], marker="o", color = 'green', s=200)
        ax.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200)
        plt.title("Map : free cells in white, occupied cells in black")

    path, visitedNodes = get_path(occupancy_grid, start, goal, movement)
    path = douglas_peucker(path, 0.8)
    if plot:
        path2 =np.array(path).reshape(-1, 2).transpose()
        print('Map with optimal path')
        fig_astar, ax_astar = create_empty_plot(occupancy_grid.shape[0], occupancy_grid.shape[1])
        ax_astar.imshow(occupancy_grid.transpose(), cmap=cmap)
        ax_astar.scatter(visitedNodes[0], visitedNodes[1], marker="o", color = 'orange')
        ax_astar.plot(path2[0], path2[1], marker="o", color = 'blue')
        ax_astar.scatter(start[0], start[1], marker="o", color = 'green', s=200)
        ax_astar.scatter(goal[0], goal[1], marker="o", color = 'purple', s=200)
    return compute_change(path)
    
    

def next_checkpoint2(path, position, counter,local_obstacle):
    if counter< len(path) -1:
        x1, y1 = path[counter]
        x2, y2 = path[counter + 1]
        x_est = x1 - position[0]
        y_est = y1 - position[1]
        dist = np.sqrt(x_est ** 2 + y_est ** 2)
        if(local_obstacle and dist<8):
            counter=counter+1
            x1, y1 = path[counter]
            x_est = x1 - position[0]
            y_est = y1 - position[1]
            dist = np.sqrt(x_est ** 2 + y_est ** 2) 
            if dist<10:  
                if counter< len(path) -1:
                    counter = counter + 1 
                x1, y1 = path[counter]
                x_est = x1 - position[0]
                y_est = y1 - position[1]
                dist = np.sqrt(x_est ** 2 + y_est ** 2) 
                if dist <12:
                    if counter< len(path) -1:
                        counter = counter + 1
                    return np.array([path[counter][0], path[counter][1]]), counter
                else : 
                    return np.array([path[counter][0], path[counter][1]]), counter
            else : 
                return np.array([path[counter][0], path[counter][1]]), counter

        if dist < 2:
            if counter< len(path) -1:
                counter = counter + 1
            return np.array([path[counter][0], path[counter][1]]), counter

    return np.array([path[counter][0], path[counter][1]]), counter
#convert a position into a cell on the grid
def convert_to_idx(position, size_cell):
    idx =[0,0]
    idx[0] = int(np.floor(position[0]/size_cell))
    idx[1] = int(np.floor(position[1]/size_cell))
    return idx

# test if the robot is on a local goal
def test_if_goal(goal, position_robot):

    if convert_to_idx(position_robot,2) == goal:
        return True
    else:
        return False





## Inutile 

def compute_change(path):
    if len(path) < 2:
        return []

    points_changement_direction = [path[0]]

    for i in range(1, len(path)-1):
        x1, y1 = path[i-1]
        x2, y2 = path[i]
        x3, y3 = path[i+1]

        pente1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        pente2 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else float('inf')

        if pente1 != pente2:
            points_changement_direction.append(path[i])

    points_changement_direction.append(path[-1])

    return points_changement_direction



def _get_movements_4n():
    """
    Get all possible 4-connectivity movements (up, down, left right).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0)]

def _get_movements_8n():
    """
    Get all possible 8-connectivity movements. Equivalent to get_movements_in_radius(1)
    (up, down, left, right and the 4 diagonals).
    :return: list of movements with cost [(dx, dy, movement_cost)]
    """
    s2 = math.sqrt(2)
    return [(1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, s2),
            (-1, 1, s2),
            (-1, -1, s2),
            (1, -1, s2)]



def reconstruct_path(cameFrom, current):
    """
    Recurrently reconstructs the path from start node to the current node
    :param cameFrom: map (dictionary) containing for each node n the node immediately 
                     preceding it on the cheapest path from start to n 
                     currently known.
    :param current: current node (x, y)
    :return: list of nodes from start to current node
    """
    total_path = [current]
    while current in cameFrom.keys():
        total_path.insert(0, cameFrom[current]) 
        current=cameFrom[current]
    return total_path

def A_Star(start, goal, h, coords, occupancy_grid, movement_type="4N"):
    """
    A* for 2D occupancy grid. Finds a path from start to goal.
    h is the heuristic function. h(n) estimates the cost to reach goal from node n.
    :param start: start node (x, y)
    :param goal_m: goal node (x, y)
    :param occupancy_grid: the grid map
    :param movement: select between 4-connectivity ('4N') and 8-connectivity ('8N', default)
    :return: a tuple that contains: (the resulting path in meters, the resulting path in data array indices)
    """
    
    
    for point in [start, goal]:
        assert point[0]>=0 and point[0]<occupancy_grid.shape[0], "start or end goal not contained in the map"
        assert point[1]>=0 and point[1]<occupancy_grid.shape[1], "start or end goal not contained in the map"
    
    if occupancy_grid[start[0], start[1]]:
        raise Exception('Start node is not traversable')

    if occupancy_grid[goal[0], goal[1]]:
        raise Exception('Goal node is not traversable')
    
    if movement_type == '4N':
        movements = _get_movements_4n()
    elif movement_type == '8N':
        movements = _get_movements_8n()
    else:
        raise ValueError('Unknown movement')
    

    openSet = [start]
    
    closedSet = []

    cameFrom = dict()

    gScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    gScore[start] = 0

    fScore = dict(zip(coords, [np.inf for x in range(len(coords))]))
    fScore[start] = h[start]

    while openSet != []:
        
        fScore_openSet = {key:val for (key,val) in fScore.items() if key in openSet}
        current = min(fScore_openSet, key=fScore_openSet.get)
        del fScore_openSet
        
        if current == goal:
            return reconstruct_path(cameFrom, current), closedSet

        openSet.remove(current)
        closedSet.append(current)
        
        for dx, dy, deltacost in movements:
            
            neighbor = (current[0]+dx, current[1]+dy)
            

            if (neighbor[0] >= occupancy_grid.shape[0]) or (neighbor[1] >= occupancy_grid.shape[1]) or (neighbor[0] < 0) or (neighbor[1] < 0):
                continue
            
            if (occupancy_grid[neighbor[0], neighbor[1]]) or (neighbor in closedSet): 
                continue
                
            
            safety_margin = calculate_safety_margin(neighbor, occupancy_grid)
            tentative_gScore = gScore[current] + deltacost*safety_margin
            
            if neighbor not in openSet:
                openSet.append(neighbor)
                
            if tentative_gScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + h[neighbor]

    print("No path found to goal")
    return [], closedSet




def get_path(occupancy_grid,start, goal, movement):
    
    x,y = np.mgrid[0:occupancy_grid.shape[0]:1, 0:occupancy_grid.shape[1]:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    pos = np.reshape(pos, (x.shape[0]*x.shape[1], 2))
    coords = list([(int(x[0]), int(x[1])) for x in pos])

    
    h = np.linalg.norm(pos - goal, axis=-1)
    h = dict(zip(coords, h))

    path, visitedNodes  = A_Star(start, goal, h , coords, occupancy_grid, movement_type=movement)
    visitedNodes = np.array(visitedNodes).reshape(-1, 2).transpose()
    return path, visitedNodes


def create_empty_plot(max_val1, max_val2):
    """
    Helper function to create a figure of the desired dimensions & grid
    
    :param max_val1: dimension of the map along the x dimension
    :param max_val2: dimension of the map along the y dimension
    :return: the fig and ax objects.
    """
    fig, ax = plt.subplots(figsize=(7,7))
    
    major_ticks1 = np.arange(0, max_val1+1, 5)
    minor_ticks1 = np.arange(0, max_val1+1, 1)
    major_ticks2 = np.arange(0, max_val2+1, 5)
    minor_ticks2 = np.arange(0, max_val2+1, 1)
    ax.set_xticks(major_ticks1)
    ax.set_xticks(minor_ticks1, minor=True)
    ax.set_yticks(major_ticks2)
    ax.set_yticks(minor_ticks2, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.set_ylim([-1,max_val2])
    ax.set_xlim([-1,max_val1])
    ax.grid(True)
    
    return fig, ax


def douglas_peucker(coords, epsilon):
    if len(coords) <= 2:
        return [coords[0], coords[-1]]

    dmax = 0
    index = 0
    end = len(coords) - 1
    for i in range(1, end):
        d = point_to_line_distance(coords[i], coords[0], coords[end])
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        results1 = douglas_peucker(coords[:index + 1], epsilon)
        results2 = douglas_peucker(coords[index:], epsilon)

        results = results1[:-1] + results2
    else:
        results = [coords[0], coords[end]]

    return results


def point_to_line_distance(point, start, end):
    numerator = abs((end[1] - start[1]) * point[0] - (end[0] - start[0]) * point[1] + end[0] * start[1] - end[1] * start[0])
    denominator = ((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2) ** 0.5
    distance = numerator / denominator if denominator != 0 else 0
    return distance

def calculate_safety_margin(neighbor, occupancy_grid):
    distance_threshold_big = 5  
    distance_threshold_small = 3
    distance_to_obstacle = distance_to_nearest_obstacle(neighbor, occupancy_grid)

    if distance_to_obstacle < distance_threshold_big:
        if distance_to_obstacle < distance_threshold_small:
            safety_margin = 2     
        else: 
            safety_margin = 1.5  
    else:
        safety_margin = 1.0  

    return safety_margin




def distance_to_nearest_obstacle(point, occupancy_grid):
   
    distances = np.linalg.norm(np.array(np.where(occupancy_grid == 1)).T - np.array(point), axis=1)
    min_distance = np.min(distances)

    return min_distance