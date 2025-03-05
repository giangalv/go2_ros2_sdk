import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq, math, random, yaml
import scipy.interpolate as si
import sys, threading, time
import random

with open("src/go2_ros2_sdk/config/params.yaml", 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

lookahead_distance = params["lookahead_distance"]
lookahead_distance_reset = params["lookahead_distance"]
speed_max = params["max_speed"]
expansion_size = params["expansion_size"]
target_error = params["target_error"]
robot_r = params["robot_r"] 
target_timeout = params["max_timeout"]

pathGlobal = 0
path_history = []
max_history_length = 4

# Euler from quaternion
def eulerFromQuaternion(x, y, z, w):
    '''
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    '''
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

# Heuristic function for A*
# Euclidean distance is used to estimate the cost to reach the goal from a to b.
def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

# A* algotihtm to find the shortest path
def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]  # Possible movement directions up, down, left, right, and diagonals
    close_set = set() # Set of visited nodes
    came_from = {} # To keep track of the path taken
    gscore = {start:0}  # Keep track of the cost to reach the goal from the start 
    fscore = {start:heuristic(start, goal)} # Keep track of the total cost of the path 
    oheap = [] # Priority queue to keep track of the nodes to visit
    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1] # The node with the lowest fscore value
        
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data = data + [start]
            data = data[::-1]
            print("[INFO] Path found:")
            return data # Return the path taken
        
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1: # 1 means obstacle
                        continue
                    
                else:
                    # array bound y walls
                    continue
                
            else:
                # array bound x walls
                continue
            
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
          
    # If no path to goal was found, return closest path to goal
    if goal not in came_from:
        closest_node = None
        closest_dist = float('inf')
        for node in close_set:
            dist = heuristic(node, goal)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist
        if closest_node is not None:
            data = []
            while closest_node in came_from:
                data.append(closest_node)
                closest_node = came_from[closest_node]
            data = data + [start]
            data = data[::-1]
            #print("[INFO] Closest path found:")
            return data
    return False


from scipy.interpolate import splprep, splev
def bsplinePlanning(array, sn):
    """
    Function to create a smooth B-spline path from a set of waypoints.

    Parameters:
    - array: np.array of shape (n_points, 2), where each row represents a point [x, y].
    - sn: (optional) number of points for spline interpolation. Default is len(array) * 5.

    Returns:
    - path: np.array of shape (sn, 2), representing the smooth path with [x, y] coordinates.
    """
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]

        if sn is None:
            sn = len(array) * 5  # Default number of interpolation points

        smoothing_factor = 0.1 # Default smoothing factor      

        # Create parameter values for t between [0, 1]
        tck, u = splprep([x, y], s=smoothing_factor, per=False)

        # Generate new interpolated points for a smooth path
        u_fine = np.linspace(0, 1, sn)
        x_smooth, y_smooth = splev(u_fine, tck)

        # Combine x and y into a single array representing the path
        path = np.vstack((x_smooth, y_smooth)).T
    except Exception as e:
        # Handle errors, print a meaningful error message, and return the original points as fallback
        print(f"BSpline planning failed: {e}")
        path = array.tolist()

    return path
    
'''
def bsplinePlanning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]

        # Determine the number of control points
        num_points = len(x)

        # Dynamically set the spline order (degree + 1)
        # Use min(3, num_points - 1) because the spline order must be less than the number of control points       
        #N = min(3, num_points - 1) 
        
        N = len(array) - 1
        if N < 2:
            N=2 
        # Generate parameter values for the control points (a simple range in this case)
        t = range(num_points)

        # Generate B-spline representations for x and y coordinates
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        # Evaluate the spline at evenly spaced points
        ipl_t = np.linspace(0.0, num_points - 1, sn)
        rx = si.splev(ipl_t, x_tup)
        ry = si.splev(ipl_t, y_tup)

        # Generate the path as a list of tuples (x, y)
        path = [(rx[i], ry[i]) for i in range(len(rx))]

    except Exception as e:
        # Handle errors, print a meaningful error message, and return the original points as fallback
        print(f"BSpline planning failed: {e}")
        path = array.tolist()

    return path
'''

def purePursuit(current_x, current_y, current_heading, path, index, lookahead_distance):
    """
    Implements pure pursuit control for a legged robot to follow a given path.

    Parameters:
    - current_x: Current x-coordinate of the robot.
    - current_y: Current y-coordinate of the robot.
    - current_heading: Current heading of the robot (in radians).
    - path: List of points representing the path as [[x1, y1], [x2, y2], ...].
    - index: Current index in the path to look for the target point.
    - lookahead_distance: The distance ahead of the robot to consider the target point.

    Returns:
    - v: Linear velocity command.
    - w: Angular velocity command.
    - new_index: Updated index of the target point on the path.
    """
    
    # Convert the path to a NumPy array for easy manipulation
    path = np.array(path)
    
    # Find the target point that is lookahead_distance away from the robot's current position
    new_index = index
    target_point_found = False
    
    while new_index < len(path):
        target_x, target_y = path[new_index]
        distance_to_point = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        
        if distance_to_point >= lookahead_distance:
            target_point_found = True
            break
        
        new_index += 1
    
    # If we reach the end of the path and no point is found, use the last point
    if not target_point_found:
        target_x, target_y = path[-1]

    # Calculate the angle to the target point
    angle_to_target = np.arctan2(target_y - current_y, target_x - current_x)

    # Calculate the heading error (how far we need to turn to face the target)
    heading_error = angle_to_target - current_heading

    # Normalize the heading error to be within the range [-pi, pi]
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    # Define control gains (these may need to be tuned based on your robot's dynamics)
    k_linear = speed_max/2  # Proportional gain for linear velocity
    k_angular = speed_max  # Proportional gain for angular velocity

    # Calculate linear velocity (based on lookahead distance and gain)
    v = k_linear * lookahead_distance
    
    #v_x = v * np.cos(current_heading)
    #v_y = v * np.sin(current_heading)

    # Calculate angular velocity (based on the heading error and gain)
    w = k_angular * heading_error

    #return v_x, v_y, w, new_index
    return v, w, new_index
'''
def purePursuit(current_x, current_y, current_heading, path, index, lookahead_distance):
    closest_point = None
    v = speed_max
    for i in range(index,len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        print("[INFO] Distance to target:", distance, "Lookahead distance:", lookahead_distance)
        if lookahead_distance < distance:
            closest_point = (x, y)
            index = i
            break
    if closest_point is not None:
        target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
        desired_steering_angle = target_heading - current_heading
    else:
        target_heading = math.atan2(path[-1][1] - current_y, path[-1][0] - current_x)
        desired_steering_angle = target_heading - current_heading
        index = len(path)-1
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi
    if desired_steering_angle > math.pi/6 or desired_steering_angle < -math.pi/6:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * math.pi/4
        v = 0.0
    return v,desired_steering_angle,index
'''

def frontierB(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0.0:
                if i > 0 and matrix[i-1][j] < 0:
                    matrix[i][j] = 2
                elif i < len(matrix)-1 and matrix[i+1][j] < 0:
                    matrix[i][j] = 2
                elif j > 0 and matrix[i][j-1] < 0:
                    matrix[i][j] = 2
                elif j < len(matrix[i])-1 and matrix[i][j+1] < 0:
                    matrix[i][j] = 2
    return matrix

def assignGroups(matrix):
    group = 1
    groups = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 2:
                group = dfs(matrix, i, j, group, groups)
    return matrix, groups

def dfs(matrix, i, j, group, groups):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
        return group
    if matrix[i][j] != 2:
        return group
    if group in groups:
        groups[group].append((i, j))
    else:
        groups[group] = [(i, j)]
    matrix[i][j] = 0
    dfs(matrix, i + 1, j, group, groups)
    dfs(matrix, i - 1, j, group, groups)
    dfs(matrix, i, j + 1, group, groups)
    dfs(matrix, i, j - 1, group, groups)
    dfs(matrix, i + 1, j + 1, group, groups) # bottom right diagonal
    dfs(matrix, i - 1, j - 1, group, groups) # top left diagonal
    dfs(matrix, i - 1, j + 1, group, groups) # top right diagonal
    dfs(matrix, i + 1, j - 1, group, groups) # bottom left diagonal
    return group + 1

def fGroups(groups):
    groups_number = 5 # Number of groups to select
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = [g for g in sorted_groups[:groups_number] if len(g[1]) > 2]    
    return top_five_groups

def calculateCentroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid

def choose_random_group(groups):
    return random.choice(list(groups.keys()))

def findClosestGroup(matrix,groups,current,resolution,originX,originY):
    targetP = None
    distances = []
    paths = []
    score = []
    max_score = -1 # max score index
    
    for i in range(len(groups)):
        middle = calculateCentroid([p[0] for p in groups[i][1]],[p[1] for p in groups[i][1]]) 
        path = astar(matrix, current, middle)
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        total_distance = pathLength(path)
        distances.append(total_distance)
        paths.append(path)
        
    for i in range(len(distances)):
        if distances[i] == 0:
            score.append(0)
        else:
            score.append(len(groups[i][1])/distances[i])
            
    for i in range(len(distances)):
        if distances[i] > target_error*3:
            if max_score == -1 or score[i] > score[max_score]:
                max_score = i
                
    if max_score != -1:
        targetP = paths[max_score]
        print("[INFO] Group selected is the closest group.")
    else: # If groups are closer than target_error*2, select a random point as the target. This helps the robot get out of some situations.
        index = random.randint(0,len(groups)-1)
        target = groups[index][1]
        target = target[random.randint(0,len(target)-1)]
        path = astar(matrix, current, target)
        targetP = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
    
    
    # Add the current path to the history
    path_history.append(targetP)

    # Keep only the last `max_history_length` paths
    if len(path_history) > max_history_length:
        path_history.pop(0)
        
    # Check if the path is repeated four times
    if len(path_history) == max_history_length and all(p == targetP for p in path_history):
        # Choose a random group if the path is repeated
        random_group_index = choose_random_group(groups)
        random_group = groups[random_group_index][1]
        random_target = random_group[random.randint(0, len(random_group) - 1)]
        path = astar(matrix, current, random_target)
        targetP = [(p[1] * resolution + originX, p[0] * resolution + originY) for p in path]
        print("[INFO] Random group selected.")
    
        # Clear the path history
        path_history.clear()
        
    return targetP

def pathLength(path):
    for i in range(len(path)):
        path[i] = (path[i][0],path[i][1])
        points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:,0], differences[:,1])
    total_distance = np.sum(distances)
    return total_distance

def costmap(data, width, height, resolution):
    data = np.array(data).reshape(height, width)
    wall = np.where(data == 100)  # Identifying obstacles
    for i, j in zip(*wall):
        x, y = np.ogrid[max(0, i-expansion_size):min(height, i+expansion_size+1),
                        max(0, j-expansion_size):min(width, j+expansion_size+1)]
        data[x, y] = 100  # Expand the obstacle
    data = data * resolution  # Apply resolution scaling
    return data


def exploration(data,width,height,resolution,column,row,originX,originY,flag_reset):
# data: 2 for frontier points, 0 for robot's current position, 1 for obstacles, and 100 for expanded obstacles
        global pathGlobal 
        if flag_reset == True:
            pathGlobal = 0
        data = costmap(data,width,height,resolution) # Expand obstacles
        data[row][column] = 0 # Robot's current position
        data[data > 5] = 1 # Set to 1 for obstacles
        data = frontierB(data) # Find frontier points
        data,groups = assignGroups(data) # Group frontier points
        groups = fGroups(groups) # Sort groups from smallest to largest. Take the top 5 largest groups
        if len(groups) == 0: # If no group, exploration is complete
            path = -1
        else: # If there is a group, find the closest group
            data[data < 0] = 1 
            path = None
            path = findClosestGroup(data,groups,(row,column),resolution,originX,originY) # Find the path to the closest group
            if path != None: # If there is a path, smooth it with BSpline
                path = bsplinePlanning(path,len(path)*5)
            else:
                path = -1
        pathGlobal = path
        return

# PROGRAM FOR in narrow spaces
def localControl(scan, random_control):
    v = None
    w = None
    
    # Define the front scan range dynamically
    mid = len(scan) // 2
    half_cone = 20 # Half of the cone angle
    front_scan_range_left = range(mid - half_cone, mid)  # Equivalent to range(104, 109) dynamically
    front_scan_range_right = range(mid, mid + half_cone)  # Equivalent to range(110, 115) dynamically
    #robot_min = robot_r - 0.1

    # Random counter to decide the order of control
    #random_counter = random.randint(0, 1)
    random_counter = random_control
    if random_counter == 0:
        # Check left first
        for i in front_scan_range_left:
            if scan[i] < robot_r:
                v = 0.0
                w = math.pi / 7 # Turn left
                break
        if w is None:
            for i in front_scan_range_right:
                if scan[i] < robot_r:
                    v = 0.0
                    w = -math.pi / 7 # Turn right
                    break
        '''
        if w is None:
            # Check the first 105 indices
            for i in range(45, mid-half_cone):  # Adjusted to 104 to match the length of the scan array
                if scan[i] < robot_min:
                    v = 0.05
                    w = math.pi / 7  # Turn left
                    break

        # Check the indices from 115 to 220
        if v is None:
            for i in range(mid+half_cone, len(scan)-45):  # Adjusted to 240 to match the length of the scan array
                if scan[i] < robot_min:
                    v = 0.05
                    w = -math.pi / 7 # Turn right
                    break
        '''
        
    else:
        # Check right first
        for i in front_scan_range_right:
            if scan[i] < robot_r:
                v = 0.0
                w = -math.pi / 7 # Turn right
                break
        if w is None:
            for i in front_scan_range_left:
                if scan[i] < robot_r:
                    v = 0.0
                    w = math.pi / 7 # Turn left
                    break
        '''
        if w is None:
            # Check the first 105 indices
            for i in range(45, mid-half_cone):  # Adjusted to 104 to match the length of the scan array
                if scan[i] < robot_min:
                    v = 0.05
                    w = -math.pi / 7  # Turn right
                    break

        # Check the indices from 115 to 220
        if v is None:
            for i in range(mid+half_cone, len(scan)-45):  # Adjusted to 240 to match the length of the scan array
                if scan[i] < robot_min:
                    v = 0.05
                    w = math.pi / 7 # Turn left
                    break
        '''
        
    return v, w 
'''    
# Local control function 
def localControl(scan):
    v = None
    w = None
    
    # Define the front scan range dynamically
    mid = len(scan) // 2
    front_scan_range_left = range(mid - 5, mid)  # Equivalent to range(104, 109) dynamically
    front_scan_range_right = range(mid, mid + 5)  # Equivalent to range(110, 115) dynamically
    
    for i in front_scan_range_left:
        if scan[i] < robot_r:
            v = 0.0
            w = math.pi / 8 # Turn left
            break
    if w is None:
        for i in front_scan_range_right:
            if scan[i] < robot_r:
                v = 0.0
                w = -math.pi / 8 # Turn right
                break
    if w is None:
        # Check the first 105 indices
        for i in range(0, mid-5):  # Adjusted to 104 to match the length of the scan array
            if scan[i] < robot_r:
                v = 0.05
                w = math.pi / 8  # Turn left
                break

        # Check the indices from 115 to 220
        if v is None:
            for i in range(mid+5, len(scan)):  # Adjusted to 240 to match the length of the scan array
                if scan[i] < robot_r:
                    v = 0.05
                    w = -math.pi / 8 # Turn right
                    break
    return v,w   
'''

class navigationControl(Node):
    def __init__(self):
        super().__init__('Exploration')
        self.lookahead_distance = lookahead_distance
        self.lookahead_distance_increment_value = 0.5
        self.lookahead_distance_max = 1.4
        self.lookahead_distance_reset = lookahead_distance
        self.flag_reset = False
        self.flag = False
        self.random_control = 0
        # Use default QoS settings for the subscription
        self.subscription = self.create_subscription(OccupancyGrid,'map',self.map_callback,10)
        self.subscription = self.create_subscription(Odometry,'odom',self.odom_callback,10)
        qos_policy=rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1, #5 works fine
        )
        self.subscription = self.create_subscription(LaserScan,'scan_obstacle',self.scan_callback,qos_profile=qos_policy)
        self.odom_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_publisher = self.create_publisher(PointStamped, 'goal_control', 10)
        print("[INFO] EXPLORATION MODE ACTIVE")
        self.exploration_mode = True
        self.timer_no_reached_goal()
        threading.Thread(target=self.exp).start() # Run the exploration function as a thread.
        
    def exp(self):
        twist = Twist()
        while True: # Wait until sensor data is received.
            if not hasattr(self,'map_data') or not hasattr(self,'odom_data') or not hasattr(self,'scan_data'):
                # Wait until the map, odometry, and scan data are received.
                time.sleep(0.1)
                continue
            if self.exploration_mode == True:
                # Set the target to the closest group of frontier points
                self.lookahead_distance = self.set_new_target()
            else:
                # Route Tracking Block Start
                v , w = localControl(self.scan, self.random_control)
                #twist.linear.y = 0.0
                if v == 0.0:
                    self.flag = True
                else:
                    if self.flag == True:
                        new_path_segment = self.path[self.i:]
                        if len(new_path_segment) > 3:  # Ensure there are enough points for B-spline planning
                            self.path = bsplinePlanning(new_path_segment, len(new_path_segment) * 3)
                            self.i = 1
                    self.flag = False
                        
                if self.flag == False:
                    v, w, self.i = purePursuit(self.x,self.y,self.yaw,self.path,self.i,self.lookahead_distance)        
                    #new_path_segment = self.path[self.i:]
                    #self.path = bsplinePlanning(new_path_segment, len(new_path_segment) * 3)
                    #self.i = 0
                                   
                if v is None:
                    v = 0.0  # Ensure v is a float
                if w is None:
                    w = 0.0  # Ensure w is a float
                
                if(abs(self.x - self.path[-1][0]) < target_error and abs(self.y - self.path[-1][1]) < target_error):
                    v = 0.0
                    w = 0.0
                    #twist.linear.y = 0.0
                    self.exploration_mode = True
                    print("[INFO] TARGET REACHED\n")
                    self.lookahead_distance = lookahead_distance_reset # Reset the lookahead distance
                    self.timer_goal.cancel()
                    self.t.join() # Wait until the thread finishes.
                    self.timer_goal
                    
                twist.linear.x = v
                twist.angular.z = w
                self.odom_publisher.publish(twist)
                time.sleep(0.1) #0.1 set the frequency of the control loop
                # Route Tracking Block End
    
    def set_new_target(self):
        if isinstance(pathGlobal, int) and pathGlobal == 0 or self.flag_reset==True:
            column = int((self.x - self.originX)/self.resolution)
            row = int((self.y - self.originY)/self.resolution)
            twist = Twist()
            # Set the v and w to 0 to stop the robot
            v = 0.0
            w = 0.0
            # Publish the v and w to the robot
            twist.linear.x = v
            #twist.linear.y = v
            twist.angular.z = w
            self.odom_publisher.publish(twist)
            time.sleep(0.01)
            exploration(self.data, self.width, self.height, self.resolution, column, row, self.originX, self.originY, self.flag_reset)
            self.flag_reset = False
            self.path = pathGlobal
        else:
            self.path = pathGlobal

        if isinstance(self.path, int) and self.path == -1:
            print("[INFO] EXPLORATION COMPLETED")
            sys.exit()
        if self.exploration_mode == True:
            self.i = 0
        self.c = int((self.path[-1][0] - self.originX)/self.resolution)
        self.r = int((self.path[-1][1] - self.originY)/self.resolution)
        self.exploration_mode = False
        
        print("[INFO] NEW TARGET SET AT: ", self.path[-1])
        goal_msg = PointStamped()
        goal_msg.header.frame_id = "map"  
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.point.x = self.path[-1][0]
        goal_msg.point.y = self.path[-1][1]
        goal_msg.point.z = 0.0  # The robot is on a 2D plane
        self.goal_publisher.publish(goal_msg)
        
        self.lookhaed_distance_increment()
        if self.random_control == 0:
            self.random_control = 1
        else:
            self.random_control = 0
        
        t = pathLength(self.path) / speed_max
        t = t - 0.2  # Subtract 0.2 seconds from the time calculated using x = v * t formula. The exploration function is called after t seconds.
        self.t = threading.Timer(t, self.target_callback)  # Call the exploration function shortly before reaching the target.
        self.timer_goal = threading.Timer((target_timeout), self.timer_no_reached_goal)
        self.t.start()
        self.timer_goal.start()
        
        return self.lookahead_distance
        

    def timer_no_reached_goal(self):
        if not hasattr(self,'map_data') or not hasattr(self,'odom_data') or not hasattr(self,'scan_data'):
            # Wait until the map, odometry, and scan data are received.
            time.sleep(0.1)
        else:
            twist = Twist()
            print("[INFO] Target not reached. Changing target.")
            # Set the v and w to 0 to stop the robot
            v = 0.0
            w = 0.0
            # Publish the v and w to the robot
            twist.linear.x = v
            #twist.linear.y = v
            twist.angular.z = w
            self.odom_publisher.publish(twist)
            # Call the method to set a new target
            self.timer_goal.cancel()
            self.t.cancel()
            self.lookahead_distance = self.set_new_target()  
    
    def lookhaed_distance_increment(self):
        self.lookahead_distance += self.lookahead_distance_increment_value
        #print("[INFO] Increase lookahead_distance to:", self.lookahead_distance)
        if self.lookahead_distance > self.lookahead_distance_max:
            self.lookahead_distance = self.lookahead_distance_reset
            self.flag_reset = True
            #print("[ERROR] Exiting.")
            #sys.exit()
        return self.lookahead_distance
         
                
    def target_callback(self):
        exploration(self.data,self.width,self.height,self.resolution,self.c,self.r,self.originX,self.originY,self.flag_reset)
        
    def scan_callback(self,msg):
        self.scan_data = msg
        self.scan = msg.ranges
        
    def map_callback(self,msg):
        self.map_data = msg
        self.resolution = self.map_data.info.resolution
        self.originX = self.map_data.info.origin.position.x
        self.originY = self.map_data.info.origin.position.y
        self.width = self.map_data.info.width
        self.height = self.map_data.info.height
        self.data = self.map_data.data

    def odom_callback(self,msg):
        self.odom_data = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = eulerFromQuaternion(msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,
                                        msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)


def main(args=None):
    rclpy.init(args=args)
    navigation_control = navigationControl()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
