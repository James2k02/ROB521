
# ===========
# ass3_q1.py
# ===========
#
# This assignment will introduce you to the idea of first building an
# occupancy grid then using that grid to estimate a robot's motion using a
# particle filter.
#
# There are two questions to complete (5 marks each):
#   Question 1: code occupancy mapping algorithm 
#   Question 2: see ass3_q2.py
#
# Fill in the required sections of this script with your code, run it to
# generate the requested plot & movie, then paste the plots into a short report
# that includes a few comments about what you've observed. Append your
# version of this script to the report. Hand in the report as a PDF file
# and the two resulting .mp4 files from Questions 1 and 2.
#
# requires: numpy, scipy, matplotlib, opencv-python, and the data file
# 'gazebo.mat'
#
# Steven Waslander March 2026


import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pathlib import Path

# Set random seed for repeatability
np.random.seed(1)

# ==========================
# Load the dataset from file
# ==========================
#    ground truth poses: t_true x_true y_true theta_true
# odometry measurements: t_odom v_odom omega_odom
#           laser scans: t_laser y_laser
#    laser range limits: r_min_laser r_max_laser
#    laser angle limits: phi_min_laser phi_max_laser

data = loadmat(r"C:\Users\james\Documents\Github\MEng\ROB521\A3\gazebo.mat")

t_true = data['t_true'].flatten()
x_true = data['x_true'].flatten()
y_true = data['y_true'].flatten()
theta_true = data['theta_true'].flatten()
t_odom = data['t_odom'].flatten()
v_odom = data['v_odom'].flatten()
omega_odom = data['omega_odom'].flatten()
t_laser = data['t_laser'].flatten()
y_laser = data['y_laser']
r_min_laser = data['r_min_laser'][0, 0]
r_max_laser = data['r_max_laser'][0, 0]
phi_min_laser = data['phi_min_laser'][0, 0]
phi_max_laser = data['phi_max_laser'][0, 0]

# =======================================
# Question 1: build an occupancy grid map
# =======================================
#
# Write an occupancy grid mapping algorithm that builds the map from the
# perfect ground-truth localization. Some of the setup is done for you
# below. The resulting map should look like "ass2_q1_soln.png". At the 
# end you will save your occupancy grid map to the file "occmap.mat" for 
# use in Question 2 of this assignment.

# Allocate a big 2D array for the occupancy grid
ogres = 0.05  # resolution of occ grid
ogxmin = -7   # minimum x value
ogxmax = 8    # maximum x value
ogymin = -3   # minimum y value
ogymax = 6    # maximum y value
ognx = int((ogxmax - ogxmin) / ogres)  # number of cells in x direction
ogny = int((ogymax - ogymin) / ogres)  # number of cells in y direction
oglo = np.zeros((ogny, ognx))  # occupancy grid in log-odds format
ogp = 0.5 * np.ones((ogny, ognx))   # occupancy grid in probability format

# Precalculate some quantities
numodom = t_odom.shape[0]
npoints = y_laser.shape[1]
angles = np.linspace(phi_min_laser, phi_max_laser, npoints)
dx = ogres * np.cos(angles)
dy = ogres * np.sin(angles)

# Interpolate the noise-free ground-truth at the laser timestamps
t_interp = np.linspace(t_true[0], t_true[numodom - 1], numodom)
x_interp = interp1d(t_interp, x_true, kind='linear')(t_laser)
y_interp = interp1d(t_interp, y_true, kind='linear')(t_laser)
theta_interp = interp1d(t_interp, theta_true, kind='linear')(t_laser)
omega_interp = interp1d(t_interp, omega_odom, kind='linear')(t_laser)

# Set up the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_size = (4 * ognx, 4 * ogny)
vid = cv2.VideoWriter('ass3_q1.mp4', fourcc, 10.0, video_size)

# Initialize figure for plotting
fig = plt.figure(1, figsize=(10, 8))
plt.clf()
plt.imshow(1 - ogp, cmap='gray', origin='lower')
plt.axis('equal')
plt.axis('off')
plt.tight_layout(pad=0)

# Convert plot to image and write first frame
fig.canvas.draw()
frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
frame_resized = cv2.resize(frame_bgr, video_size)
vid.write(frame_resized)

# ------insert your occupancy grid mapping parameters here------

# Log-odds update parameters for the inverse sensor model
log_odds_occ = 0.85
log_odds_free = -0.4
log_odds_min = -5.0
log_odds_max = 5.0

# ------end of your occupancy grid mapping algorithm-------


# Loop over laser scans (every fifth)
for i in range(0, t_laser.shape[0], 5):
    
    # ------insert your occupancy grid mapping algorithm here------
    # Get current robot pose
    x = (x_interp[i] - ogxmin) / ogres
    y = (y_interp[i] - ogymin) / ogres

    # Loop over each laser scan point
    for j in range(npoints):

        # Check valid laser range
        if r_min_laser <= y_laser[i, j] <= r_max_laser:

            # Convert range to grid units
            range_pixel = y_laser[i, j] / ogres

            # Compute laser angle
            theta_laser = theta_interp[i] + angles[j]

            # Normalize angle to [-pi, pi]
            theta_laser = np.arctan2(np.sin(theta_laser), np.cos(theta_laser))

            # Endpoint of ray
            x_end = int(round(x + range_pixel * np.cos(theta_laser)))
            y_end = int(round(y + range_pixel * np.sin(theta_laser)))

            # Initialize ray indices
            x_idxs = []
            y_idxs = []

            x_step = x
            y_step = y

            # Step along the ray
            for step in range(1, int(np.ceil(range_pixel)) + 1):

                x_step = int(round(x + step * np.cos(theta_laser)))
                y_step = int(round(y + step * np.sin(theta_laser)))

                # Stop if out of bounds
                if x_step <= 0 or x_step > ognx or y_step <= 0 or y_step > ogny:
                    break

                x_idxs.append(x_step)
                y_idxs.append(y_step)

            # Update free space (decrease log-odds)
            for k in range(len(x_idxs) - 1):
                if (0 < x_idxs[k] <= ognx) and (0 < y_idxs[k] <= ogny):
                    oglo[y_idxs[k], x_idxs[k]] -= 0.5

            # Update occupied cell (endpoint)
            if (0 < x_end <= ognx) and (0 < y_end <= ogny):
                oglo[y_end, x_end] += 1.5


    # Convert log-odds to probability
    ogp = 1 - 1 / (1 + np.exp(oglo))

    # ------end of your occupancy grid mapping algorithm-------
    
    # Draw the map
    plt.clf()
    plt.imshow(1 - ogp, cmap='gray', origin='lower')
    plt.axis('equal')
    plt.axis('off')
    
    # Draw the robot
    x = (x_interp[i] - ogxmin) / ogres
    y = (y_interp[i] - ogymin) / ogres
    th = theta_interp[i]
    r = 0.15 / ogres
    
    # Draw robot as a circle with heading indicator
    circle = patches.Circle((x, y), r, linewidth=2, edgecolor='blue', 
                           facecolor=[0.35, 0.35, 0.75])
    plt.gca().add_patch(circle)
    
    # Draw heading line
    plt.plot([x, x + r * np.cos(th)], [y, y + r * np.sin(th)], 'k-', linewidth=2)
    
    # Save the video frame
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame_bgr, video_size)
    vid.write(frame_resized)
    
    plt.pause(0.1)

vid.release()
plt.savefig('ass3_q1_result.png', dpi=100, bbox_inches='tight')
plt.close()

# Save the occupancy grid map
import scipy.io as sio
sio.savemat('occmap.mat', {
    'ogres': ogres,
    'ogxmin': ogxmin,
    'ogxmax': ogxmax,
    'ogymin': ogymin,
    'ogymax': ogymax,
    'ognx': ognx,
    'ogny': ogny,
    'oglo': oglo,
    'ogp': ogp
})

print("Occupancy grid mapping complete!")
print(f"Video saved as: ass3_q1.mp4")
print(f"Map image saved as: ass3_q1_result.png")
print(f"Occupancy grid data saved as: occmap.mat")
