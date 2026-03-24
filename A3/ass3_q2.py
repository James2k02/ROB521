# =========
# ass3_q2.m
# =========
#
# This assignment will introduce you to the idea of first building an
# occupancy grid then using that grid to estimate a robot's motion using a
# particle filter.
# 
# There are two questions to complete (5 marks each):
#
#    Question 1: see ass3_q1.m 
#    Question 2: code particle filter to localize from known map
#
# Fill in the required sections of this script with your code, run it to
# generate the requested plot & movie, then paste the plots into a short report
# that includes a few comments about what you've observed.  Append your
# version of this script to the report.  Hand in the report as a PDF file
# and the two resulting .mp4 files from Questions 1 and 2.
#
# requires: numpy, scipy, matplotlib, opencv-python, and the data files
# 'gazebo.mat', 'occmap.mat'
#
# Steven Waslander March 2026
#

import math

import cv2
import numpy as np
from scipy.io import loadmat

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def mat_scalar(value):
    return float(np.asarray(value).squeeze())


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def canvas_to_bgr(fig, frame_size):
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.resize(frame_bgr, frame_size)


def draw_robot(ax, x_world, y_world, heading, cell_size, x_min, y_min):
    x_cell = (x_world - x_min) / cell_size
    y_cell = (y_world - y_min) / cell_size
    robot_radius = 0.15 / cell_size
    robot = patches.Circle(
        (x_cell, y_cell),
        robot_radius,
        linewidth=2,
        edgecolor="black",
        facecolor=[0.35, 0.35, 0.75],
    )
    ax.add_patch(robot)
    ax.plot(
        [x_cell, x_cell + robot_radius * math.cos(heading)],
        [y_cell, y_cell + robot_radius * math.sin(heading)],
        "k-",
        linewidth=2
    )


# =========================================================================
# Question 2: localization from an occupancy grid map using particle filter
# =========================================================================
#
# Write a particle filter localization algorithm to localize from the laser
# rangefinder readings, wheel odometry, and the occupancy grid map you
# built in Question 1.  We will only use two laser scan lines at the
# extreme left and right of the field of view, to demonstrate that the
# algorithm does not need a lot of information to localize fairly well.  To
# make the problem harder, the below lines add noise to the wheel odometry
# and to the laser scans.  The plot "ass2_q2_soln.png" shows
# the errors in the estimates produced by wheel odometry alone and by the
# particle filter look like as compared to ground truth; we can see that
# the errors are much lower when we use the particle filter.


def main():
    # ---- toggle display and recordings ----
    SHOW_PLOT = True      # show live plot window during run
    RECORD_VIDEO = False   # write frames to video file
    # ---------------------------------------

    np.random.seed(1)
    if SHOW_PLOT:
        plt.ion()

    # load the sensor and map data
    gazebo = loadmat("gazebo.mat")
    occmap = loadmat("occmap.mat")

    t_true = gazebo["t_true"].flatten()
    x_true = gazebo["x_true"].flatten()
    y_true = gazebo["y_true"].flatten()
    theta_true = gazebo["theta_true"].flatten()
    t_odom = gazebo["t_odom"].flatten()
    v_odom = gazebo["v_odom"].flatten()
    omega_odom = gazebo["omega_odom"].flatten()
    t_laser = gazebo["t_laser"].flatten()
    y_laser = gazebo["y_laser"].astype(float)
    phi_min_laser = mat_scalar(gazebo["phi_min_laser"])
    phi_max_laser = mat_scalar(gazebo["phi_max_laser"])
    r_max_laser = mat_scalar(gazebo["r_max_laser"])

    ogres = mat_scalar(occmap["ogres"])
    ogxmin = mat_scalar(occmap["ogxmin"])
    ogxmax = mat_scalar(occmap["ogxmax"])
    ogymin = mat_scalar(occmap["ogymin"])
    ogymax = mat_scalar(occmap["ogymax"])
    ogp = occmap["ogp"].astype(float)

    # interpolate the noise-free ground-truth at the laser timestamps
    numodom = t_odom.shape[0]
    t_interp = np.linspace(t_true[0], t_true[numodom - 1], numodom)
    x_interp = np.interp(t_laser, t_interp, x_true)
    y_interp = np.interp(t_laser, t_interp, y_true)
    theta_interp = np.interp(t_laser, t_interp, theta_true)

    # interpolate the wheel odometry at the laser timestamps and
    # add noise to measurements (yes, on purpose to see effect)
    v_interp = np.interp(t_laser, t_interp, v_odom) + 0.2 * np.random.randn(t_laser.shape[0])
    omega_interp = np.interp(t_laser, t_interp, omega_odom) + 0.04 * np.random.randn(t_laser.shape[0])

    # add noise to the laser range measurements (yes, on purpose to see effect)
    # and precompute some quantities useful to the laser
    y_laser = y_laser + 0.1 * np.random.randn(*y_laser.shape)
    npoints = y_laser.shape[1]
    angles = np.linspace(phi_min_laser, phi_max_laser, npoints)
    y_laser_max = 5.0

    # particle filter tuning parameters (yours may be different)
    nparticles = 200
    v_noise = 0.2
    u_noise = 0.2
    omega_noise = 0.04
    laser_var = 0.5 ** 2
    w_gain = 10.0 * math.sqrt(2.0 * math.pi * laser_var)

    # generate an initial cloud of particles
    x_particle = x_true[0] + 0.5 * np.random.randn(nparticles)
    y_particle = y_true[0] + 0.3 * np.random.randn(nparticles)
    theta_particle = theta_true[0] + 0.1 * np.random.randn(nparticles)

    # initialize the odom solution
    x_odom_only = x_true[0]
    y_odom_only = y_true[0]
    theta_odom_only = theta_true[0]

    pf_err = np.zeros_like(t_laser, dtype=float)
    wo_err = np.zeros_like(t_laser, dtype=float)

    height, width = ogp.shape
    frame_size = (4 * width, 4 * height)

    if RECORD_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter("ass2_q2.mp4", fourcc, 10.0, frame_size)
    else:
        video = None

    if SHOW_PLOT or RECORD_VIDEO:
        fig_map, ax_map = plt.subplots(figsize=(10, 8))
        if SHOW_PLOT:
            fig_map.show()
    else:
        fig_map, ax_map = None, None

    # update the output frae to include the map, robot, measurements, filter particles and odom solution
    def render_map_frame(scan_index):
        if not SHOW_PLOT and not RECORD_VIDEO:
            return

        # Display the map
        ax_map.clear()
        ax_map.imshow(1.0 - ogp, cmap="gray", origin="lower")
        ax_map.scatter(
            (x_particle - ogxmin) / ogres,
            (y_particle - ogymin) / ogres,
            s=10,
            c=[[0.0, 0.6, 0.0]],
        )
        # Display the odometry-only solution
        ax_map.scatter(
            [(x_odom_only - ogxmin) / ogres],
            [(y_odom_only - ogymin) / ogres],
            s=60,
            c="red",
        )

        # Display the true robot position and laser beams
        x_true_cell = (x_interp[scan_index] - ogxmin) / ogres
        y_true_cell = (y_interp[scan_index] - ogymin) / ogres
        ax_map.scatter(
            [(np.mean(x_particle) - ogxmin) / ogres],
            [(np.mean(y_particle) - ogymin) / ogres],
            s=60,
            c="green",
        )

        # Display the laser beam measurements
        for beam_index in (0, npoints - 1):
            beam_range = y_laser[scan_index, beam_index]
            if np.isfinite(beam_range) and beam_range <= y_laser_max:
                ax_map.plot(
                    [x_true_cell, x_true_cell + beam_range / ogres * math.cos(theta_interp[scan_index] + angles[beam_index])],
                    [y_true_cell, y_true_cell + beam_range / ogres * math.sin(theta_interp[scan_index] + angles[beam_index])],
                    "m-",
                    linewidth=1,
                )

        # Display the robot
        draw_robot(ax_map, x_interp[scan_index], y_interp[scan_index], theta_interp[scan_index], ogres, ogxmin, ogymin)
        ax_map.set_aspect("equal")
        ax_map.axis("off")
        fig_map.tight_layout(pad=0)

        if SHOW_PLOT:
            fig_map.canvas.draw_idle()
            fig_map.canvas.flush_events()
            plt.pause(0.001)

        if RECORD_VIDEO:
            video.write(canvas_to_bgr(fig_map, frame_size))

    render_map_frame(0)

    beam_indices = (0, npoints - 1)
    normalizer = math.sqrt(2.0 * math.pi * laser_var)

    # Main filter loop
    for i in range(1, t_laser.shape[0]):
        dt = t_laser[i] - t_laser[i - 1]

        v = v_interp[i]
        omega = omega_interp[i]
        x_odom_only = x_odom_only + dt * v * math.cos(theta_odom_only)
        y_odom_only = y_odom_only + dt * v * math.sin(theta_odom_only)
        theta_odom_only = wrap_to_pi(theta_odom_only + dt * omega)

        w_particle = np.ones(nparticles, dtype=float)

        # Loop over particles
        for n in range(nparticles):
            
            # propagate the particle forward in time using wheel odometry
            v_sample = v_interp[i] + v_noise * np.random.randn()
            u_sample = u_noise * np.random.randn()
            omega_sample = omega_interp[i] + omega_noise * np.random.randn()

            x_particle[n] = x_particle[n] + dt * (
                v_sample * math.cos(theta_particle[n]) - u_sample * math.sin(theta_particle[n])
            )
            y_particle[n] = y_particle[n] + dt * (
                v_sample * math.sin(theta_particle[n]) + u_sample * math.cos(theta_particle[n])
            )
            theta_particle[n] = wrap_to_pi(theta_particle[n] + dt * omega_sample)

            for beam_index in beam_indices:
                measured_range = y_laser[i, beam_index]
                if not np.isfinite(measured_range) or measured_range > y_laser_max or measured_range <= 0.0:
                    continue

                # ------insert your particle filter weight calculation here ------
















                # ------end of your particle filter weight calculation-------

        weight_sum = np.sum(w_particle)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            w_particle.fill(1.0 / nparticles)
        else:
            w_particle /= weight_sum

        cumulative_weights = np.cumsum(w_particle)
        start = np.random.rand() / nparticles
        targets = start + np.arange(nparticles) / nparticles
        indices = np.searchsorted(cumulative_weights, targets, side="left")
        indices = np.clip(indices, 0, nparticles - 1)

        x_particle = x_particle[indices]
        y_particle = y_particle[indices]
        theta_particle = theta_particle[indices]

        pf_err[i] = math.sqrt((np.mean(x_particle) - x_interp[i]) ** 2 + (np.mean(y_particle) - y_interp[i]) ** 2)
        wo_err[i] = math.sqrt((x_odom_only - x_interp[i]) ** 2 + (y_odom_only - y_interp[i]) ** 2)

        render_map_frame(i)

    if RECORD_VIDEO:
        video.release()
    if SHOW_PLOT or RECORD_VIDEO:
        plt.close(fig_map)

    fig_err, ax_err = plt.subplots(figsize=(8, 4.5))
    ax_err.plot(t_laser, pf_err, "g-", label="particle filter")
    ax_err.plot(t_laser, wo_err, "r-", label="odom")
    ax_err.set_xlabel("t [s]")
    ax_err.set_ylabel("error [m]")
    ax_err.set_title("error (estimate-true)")
    ax_err.legend(loc="upper left")
    fig_err.tight_layout()
    fig_err.savefig("ass2_q2.png", dpi=150)
    plt.close(fig_err)

    print("Particle filter localization complete!")
    if RECORD_VIDEO:
        print("Video saved as: ass2_q2.mp4")
    print("Error plot saved as: ass2_q2.png")


if __name__ == "__main__":
    main()