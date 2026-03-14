
# ======
# ROB521_assignment2.py
# ======
#
# This assignment will introduce you to the idea of estimating the motion 
# of a mobile robot using wheel odometry, and then also using that wheel
# odometry to make a simple map.  It uses a dataset previously gathered in
# a mobile robot simulation environment called Gazebo. Watch the video,
# 'gazebo.mp4' to visualize what the robot did, what its environment
# looks like, and what its sensor stream looks like.
# 
# There are three questions to complete (5 marks each):
#
#    Question 1: code (noise-free) wheel odometry algorithm
#    Question 2: add noise to data and re-run wheel odometry algorithm
#    Question 3: build a map from ground truth and noisy wheel odometry
#
# Fill in the required sections of this script with your code, run it to
# generate the requested plots, then paste the plots into a short report
# that includes a few comments about what you've observed.  Append your
# version of this script to the report.  Hand in the report as a PDF file.


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

np.random.seed(1)

# ==========================
# load the dataset from file
# ==========================
#
#    ground truth poses: t_true x_true y_true theta_true
#    laser scans: t_laser y_laser
#    laser range limits: r_min_laser r_max_laser
#    laser angle limits: phi_min_laser phi_max_laser

# load data (assumes the .mat file is in the same folder)
data = loadmat(r"C:\Users\james\OneDrive\Documents\University\MEng\ROB521\A2\ROB521_assignment2_gazebo_data.mat") # r"A2\ROB521_assignment2_gazebo_data.mat")

# helper to squeeze Matlab arrays to 1D where appropriate
squeeze = lambda a: np.array(a).squeeze()

t_true = squeeze(data['t_true'])
x_true = squeeze(data['x_true'])
y_true = squeeze(data['y_true'])
theta_true = squeeze(data['theta_true'])

t_odom = squeeze(data['t_odom'])
v_odom = squeeze(data['v_odom'])
omega_odom = squeeze(data['omega_odom'])

# laser
t_laser = squeeze(data['t_laser'])
y_laser = np.array(data['y_laser'])

r_min_laser = float(squeeze(data['r_min_laser']))
r_max_laser = float(squeeze(data['r_max_laser']))
phi_min_laser = float(squeeze(data['phi_min_laser']))
phi_max_laser = float(squeeze(data['phi_max_laser']))

# ----------------
# Question 1: noise-free wheel odometry
# ----------------

# Write an algorithm to estimate the pose of the robot throughout motion
# using the wheel odometry data (t_odom, v_odom, omega_odom) and assuming
# (x_odom y_odom theta_odom) so that the comparison plots can be generated
# below.  See the plot 'ass2_q1_soln.png' for what your results should look
# like.

# variables to store wheel odometry pose estimates
numodom = t_odom.size
x_odom = np.zeros(numodom)
y_odom = np.zeros(numodom)
theta_odom = np.zeros(numodom)

# initialize from ground truth
x_odom[0] = x_true[0]
y_odom[0] = y_true[0]
theta_odom[0] = theta_true[0]

# ------insert your wheel odometry algorithm here-------

for i in range(1, numodom):
    # For each odom measurement, calculate change in x, y, theta (just propogate using the velocity input and add it to the previous state)
    dt = t_odom[i] - t_odom[i-1]

    x_odom[i] = x_odom[i-1] + v_odom[i-1] * np.cos(theta_odom[i-1]) * dt
    y_odom[i] = y_odom[i-1] + v_odom[i-1] * np.sin(theta_odom[i-1]) * dt
    theta_odom[i] = theta_odom[i-1] + omega_odom[i-1] * dt

# ------end of your wheel odometry algorithm-------


# plots
fig, axs = plt.subplots(2,2, figsize=(10,8))
axs = axs.flatten()
axs[0].plot(x_true, y_true, 'b')
axs[0].plot(x_odom, y_odom, 'r')
axs[0].legend(['true','odom'])
axs[0].set_xlabel('x [m]')
axs[0].set_ylabel('y [m]')
axs[0].set_title('path')
axs[0].axis('equal')

# wrap heading to [-pi,pi]
theta_odom = np.arctan2(np.sin(theta_odom), np.cos(theta_odom))
axs[1].plot(t_true, theta_true, 'b')
axs[1].plot(t_odom, theta_odom, 'r')
axs[1].legend(['true','odom'])
axs[1].set_xlabel('t [s]')
axs[1].set_ylabel('theta [rad]')
axs[1].set_title('heading')

pos_err = np.sqrt((x_odom - x_true)**2 + (y_odom - y_true)**2)
axs[2].plot(t_odom, pos_err, 'b')
axs[2].set_xlabel('t [s]')
axs[2].set_ylabel('distance [m]')
axs[2].set_title('position error (odom-true)')

# wrap angular error to [-pi,pi]
theta_err = np.arctan2(np.sin(theta_odom - theta_true), np.cos(theta_odom - theta_true))
axs[3].plot(t_odom, theta_err, 'b')
axs[3].set_xlabel('t [s]')
axs[3].set_ylabel('theta [rad]')
axs[3].set_title('heading error (odom-true)')

plt.tight_layout()
plt.show(block=False)
fig.savefig(r"C:\Users\james\OneDrive\Documents\University\MEng\ROB521\A2\ass2_q1.png", dpi=150)

# =================================================================
# Question 2: add noise to data and re-run wheel odometry algorithm
# =================================================================
#
# Now we're going to deliberately add some noise to the linear and 
# angular velocities to simulate what real wheel odometry is like.  Copy
# your wheel odometry algorithm from above into the indicated place below
# to see what this does.  The below loops 100 times with different random
# noise.  See the plot 'ass2_q2_soln.png' for what your results should look
# like.

# save the original odometry variables for later use
v_odom_noisefree = np.copy(v_odom)
omega_odom_noisefree = np.copy(omega_odom)

fig2 = plt.figure(figsize=(6,6))
ax = fig2.add_subplot(1,1,1)

# loop multiple random trials
for n in range(100):
    
    # add noise to wheel odometry measurements (yes, on purpose to see effect)
    v_odom = v_odom_noisefree + 0.2 * np.random.randn(numodom)
    omega_odom = omega_odom_noisefree + 0.04 * np.random.randn(numodom)

    # ------insert your wheel odometry algorithm here-------
    for i in range(1, numodom):
        # For each odom measurement, calculate change in x, y, theta (just propogate using the velocity input and add it to the previous state)
        dt = t_odom[i] - t_odom[i-1]

        x_odom[i] = x_odom[i-1] + v_odom[i-1] * np.cos(theta_odom[i-1]) * dt
        y_odom[i] = y_odom[i-1] + v_odom[i-1] * np.sin(theta_odom[i-1]) * dt
        theta_odom[i] = theta_odom[i-1] + omega_odom[i-1] * dt
        
    # ------end of your wheel odometry algorithm------- 
  
    ax.plot(x_odom, y_odom, color='r', alpha=0.15)

# plot ground truth on top
ax.plot(x_true, y_true, 'b')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('path')
ax.axis('equal')

plt.show(block=False)
fig2.savefig(r"C:\Users\james\OneDrive\Documents\University\MEng\ROB521\A2\ass2_q2.png", dpi=150)

# After this loop, v_odom and omega_odom remain as the last noisy trial and can be used in the next question 
# to build a map from noisy odometry.  You can also re-run this loop with different noise levels to see how 
# it affects the results.

# ----------------
# Question 3: build map from laser scans using noisy and noise-free poses
# ----------------
# Now we're going to try to plot all the points from our laser scans in the
# robot's initial reference frame.  This will involve first figuring out
# how to plot the points in the current frame, then transforming them back
# to the initial frame and plotting them.  Do this for both the ground
# truth pose (blue) and also the last noisy odometry that you calculated in 
# Question 2 (red).  At first even the map based on the ground truth may
# not look too good.  This is because the laser timestamps and odometry
# timestamps do not line up perfectly and you'll need to interpolate.  Even
# after this, two additional patches will make your map based on ground
# truth look as crisp as the one in 'ass2_q3_soln.png'.  The first patch is
# to only plot the laser scans if the angular velocity is less than 
# 0.1 rad/s; this is because the timestamp interpolation errors have more
# of an effect when the robot is turning quickly.  The second patch is to
# account for the fact that the origin of the laser scans is about 10 cm
# behind the origin of the robot.  Once your ground truth map looks crisp,
# compare it to the one based on the odometry poses, which should be far
# less crisp, even with the two patches applied.

fig3 = plt.figure(figsize=(8,8))
ax3 = fig3.add_subplot(1,1,1)

# Determining the laser beam angles
npoints = y_laser.shape[1] # y_laser is a 2D array that is [num_laser_scans, num_points_per_scan], so this gets the number of points per scan
angles = np.linspace(phi_min_laser, phi_max_laser, npoints) # generates angles for each beam laser scan, from phi_min_laser to phi_max_laser, with npoints total points

# Loop over both the noisy odometry and the ground truth to build two maps on the same plot
for n in [1,2]:
    if n == 1:
        # noisy odometry at laser timestamps (use last noisy odometry arrays)
        x_interp = np.interp(t_laser, t_odom, x_odom)
        y_interp = np.interp(t_laser, t_odom, y_odom)
        # unwrap theta before interpolation to avoid jumps
        #theta_unwrapped = np.unwrap(theta_odom)
        theta_interp = np.interp(t_laser, t_odom, theta_odom)
        omega_interp = np.interp(t_laser, t_odom, omega_odom)
        color = 'r'
    else:
        # ground truth interpolation at laser timestamps
        x_interp = np.interp(t_laser, t_true, x_true)
        y_interp = np.interp(t_laser, t_true, y_true)
        #theta_unwrapped = np.unwrap(theta_true)
        theta_interp = np.interp(t_laser, t_true, theta_true)
        omega_interp = np.interp(t_laser, t_true, omega_odom_noisefree)
        color = 'b'
    
    # loop over laser scans
    for i in range(t_laser.size):
        # ------insert your point transformation algorithm here------
        # Laser scans and noisy odometry are now aligned.
        # Transform the laser scans into the current robot frame.
        # The laser is 10 cm behind the robot.
        # Only plot for low rotational velocities to avoid interpolation errors.

        if abs(omega_interp[i]) < 0.1: # skip if robot is turning quickly to avoid interpolation errors

            # Transform laser scans into current robot frame (converting from polar to cartesian coordinates, and accounting for the 10 cm offset of the laser from the robot's origin)
            laser_curr_robo_x = (y_laser[i, :] + 0.1) * np.cos(angles) # y_laser[i,:] is all the range measurements for the ith laser scan
            laser_curr_robo_y = (y_laser[i, :] + 0.1) * np.sin(angles)

            # Transform current frame laser scans into initial robot frame (expanded version of the rotation matrix transformation, then add the current position)
            # x_world = x_robot + x_laser * cos(theta_robot) - y_laser * sin(theta_robot)
            # y_world = y_robot + x_laser * sin(theta_robot) + y_laser * cos(theta_robot)
            laser_initial_robo_x = (laser_curr_robo_x * np.cos(theta_interp[i]) - laser_curr_robo_y * np.sin(theta_interp[i]) + x_interp[i])
            laser_initial_robo_y = (laser_curr_robo_x * np.sin(theta_interp[i]) + laser_curr_robo_y * np.cos(theta_interp[i]) + y_interp[i])

            # Plot the points
            if n == 1:
                plt.scatter(laser_initial_robo_x, laser_initial_robo_y, s = 6, c = 'r')
            else: 
                plt.scatter(laser_initial_robo_x, laser_initial_robo_y, s = 6, c = 'b')
        
    
        # ------end of your point transformation algorithm-------
        
        
ax3.set_aspect('equal')
plt.show()

# save each figure to disk
fig3.savefig(r"C:\Users\james\OneDrive\Documents\University\MEng\ROB521\A2\ass2_q3.png", dpi=150)

print('Saved ass2_q1.png, ass2_q2.png, ass2_q3.png')