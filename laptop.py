"""
Copyright (c) 2026 The uos_imr_build Authors.
Authors: Blair Thornton, Sam Fenton, Miquel Massot 

All rights reserved. Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import numpy as np
import argparse
from datetime import datetime
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import RBLaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate

#-------------- Guided-practicals -------------------------------
# IMR_RF2_T1_IMPORTS: Add the LiDAR and motion-model imports here
# IMR_RF2_T2_IMPORTS: Add the freespace and steering imports here
# IMR_RF2_T3_IMPORTS: Add the forward-clearance and braking imports here
#----------------------------------------------------------------
# IMR_P2_T3_IMPORTS: Add the motion-model imports here
from model_uos_imr import ActuatorConfiguration
from math_uos_imr import Vector
from model_uos_imr import rigid_body_kinematics
# IMR_P2_T4_IMPORTS: Add the LiDAR observation-model and Vector imports here
from model_uos_imr import RangeAngleKinematics
#----------------------------------------------------------------
# IMR_P3_T1_IMPORTS: Add trajectory-generation imports here
from model_uos_imr import TrajectoryGenerate
from math_uos_imr import l2m
# IMR_P3_T2_IMPORTS: Add feedback-control and frame-transformation imports here
from model_uos_imr import feedback_control
from math_uos_imr import Inverse, HomogeneousTransformation
#----------------------------------------------------------------
# IMR_P7_T1_IMPORTS: Add the LiDAR and range-angle model imports here.
# IMR_P7_T2_IMPORTS: Add the GPC classifier imports here.
# IMR_P7_T3_IMPORTS: Add the GPR and particle-filter imports here.
#----------------------------------------------------------------

class LaptopPilot:
    def __init__(self, simulation):
        
        #-------------- Guided-practicals -------------------------------
        # IMR_RF2_T1_ARUCO: Set the arena port and robot marker ID here.
        # IMR_P2_T1_ARUCO: Set the arena port and robot marker ID here.
        #----------------------------------------------------------------
        aruco_params = {
            "port": 50001,  # Port to listen Arena1: 50001; Arena2: 50002 (CHANGE THIS to match the Arena you are testing in)
            "marker_id": 20,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
        }
        self.robot_ip = "192.168.90.1" # Don't change this        

        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True) #Auto-create folder if missing                

        # handles different time reference, network amd aruco parameters for simulator
        self.sim_time_offset = 0 #used to deal with webots timestamps
        self.sim_init = False #used to deal with webots timestamps
        self.simulation = simulation
        self.start_time = None
        self.path = None
        if self.simulation:
            aruco_params = {
                "port": 50000,  # Port to listen to (DO NOT CHANGE)
                "marker_id": 0,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
            }
            self.robot_ip = "127.0.0.1"          
            aruco_params['marker_id'] = 0  # Overrides the ArUco marker ID for simulation

            #Creates a fake Aruco log
            filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")            
            self.groundtruth_log = log_dir / f"{filename_time}_pseudo_aruco.csv"
            with self.groundtruth_log.open('w') as f:
                f.write("epoch [s],elapsed [s],x [m],y [m],z [m],roll [deg],pitch [deg],yaw [deg],broadcast\n")
            self.sim_init = True #used to deal with webots timestamps
            
        self.broadcast = None # stores aruco broadcast timestamps to flag assignment in pseudo aruco (simulated) data. Not used for physical system
        self.pending_groundtruth = None   # holds the last aruco sample until the next callback. Only used for simulation. Not used for physical system
        self.stop_flag = False # a flag to stop wheels when code ends
        
        print("Connecting to robot with IP", self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)

        ############# INITIALISE ATTRIBUTES ##########        
        
        #-------------- Guided-practicals -------------------------------
        # IMR_P3_T1_WAYPOINTS: Define the northing and easting waypoint
        # lists and specify whether they are relative to the initial position
        #----------------------------------------------------------------
        # path
        self.northings_path = [0, 1, 1, 2, 2] # create a list of waypoints
        self.eastings_path = [0, 0, 1, 1, 0] # create a list of waypoints
        self.relative_path = True # False if you want it to be absolute
        #-------------- Guided-practicals -------------------------------
        # IMR_RF2_T1_ACTUATORS: Set actuation configuration parameters
        # IMR_RF2_T1_MOTION_INIT: Add the pose-initialisation flag here
        # IMR_RF2_T2_SET_TWIST: Set twist parameters
        # IMR_RF2_T3_SET_TWIST: Set twist parameters
        #----------------------------------------------------------------
        # IMR_P2_T3_INITIALISATION: Add the pose-initialisation flag here
        self.initialise_pose = True # False once the pose is initialised 
        # IMR_P2_T3_ACTUATOR_MODEL: Define the wheel geometry and create
        # the differential-drive actuator model here.
        # modelling parameters
        wheel_distance = 0.081 # measure this 
        wheel_diameter = 0.074 # measure this
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) #look at your tutorial and see how to use this
        #----------------------------------------------------------------
        # IMR_P3_T1_TRAJECTORY_PARAMETERS: Define the velocity,
        # acceleration, waypoint acceptance radius and turning radius here.
        self.v = 0.1
        self.a = 0.1/3
        self.accept_radius = 0.2
        self.arc_radius = 0.3
        # IMR_P3_T2_CONTROL_PARAMETERS: Define the controller response
        # parameters, motion limits and control-initialisation flag here.
        # control parameters        
        self.tau_s = 1 # s to remove along track error
        self.L = 0.5 # m distance to remove normal and angular error
        self.v_max = 0.2 # fastest the robot can go
        self.w_max = np.deg2rad(30) # fastest the robot can turn
        self.initialise_control = True # False once control gains is initialised 
        # IMR_P3_T3_PARAMETERS: Change the trajectory and control
        # parameters above when investigating parameter sensitivity.
        #----------------------------------------------------------------

        # model pose
        self.est_pose_northings_m = 0
        self.est_pose_eastings_m = 0
        self.est_pose_yaw_rad = 0

        # measured pose
        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        # wheel speed commands
        self.cmd_wheelrate_right = None
        self.cmd_wheelrate_left = None 

        # encoder/actual wheel speeds
        self.measured_wheelrate_right = None
        self.measured_wheelrate_left = None   

        # lidar
        self.lidar_timestamp_s = None
        self.lidar_data = None        
        self.corners = [] # list of corners locations. Append to this using self.corners.append([n,e])        

        #-------------- Guided-practicals -------------------------------
        # IMR_RF2_T1_LIDAR_MODEL: Define the LiDAR position and create the
        # range-angle observation model here
        # IMR_RF2_T2_FREESPACE_PARAMS: Set freespace and steering 
        # method parameters here
        # IMR_RF2_T3_BRAKING_PARAMS: Set braking method parameters here
        # IMR_RF2_T4_ESCAPE_PARAMS: Set escape method parameters here
        #----------------------------------------------------------------
        # IMR_P2_T4_LIDAR_MODEL: Define the LiDAR position and create the
        # range-angle observation model here
        # lidar       
        lidar_xb = 0.04 # location of lidar centre in b-frame primary axis
        lidar_yb = 0 # location of lidar centre in b-frame secondary axis
        self.lidar = RangeAngleKinematics(lidar_xb,lidar_yb, distance_range = [0.05, 1], scan_fov = np.deg2rad(120), n_beams = 20)
        #----------------------------------------------------------------
        
        ###############################################################
        # Outputs used by show_laptop.py for GUI visualisation
        #
        # state_estimation_method:
        #   1 = EKF
        #   2 = Particle Filter
        #   3 = GraphSLAM
        #   4 = Map-based Particle SLAM
        #
        # Additional visualisations:
        #   self.freespace_vector: Reactive navigation        
        ###############################################################
        self.state_estimation_method = None
        self.cov = None              # method 1: state covariance
        self.particles = None        # method 2: particle cloud
        self.graph = None            # method 3: graph
        self.particle_filter = None  # method 4: particles, weights and maps
        self.freespace_vector = None # Reactive control vector, shape (2, 1)
        ###############################################################

        #-------------- Guided-practicals -------------------------------
        # IMR_P7_T1_LIDAR_MODEL: Define the LiDAR position, create the
        # range-angle model, initialise range-bearing scan storage and set
        # the new-scan flag to False here
        # IMR_P7_T2_GPC_SETUP: Load the trained corner classifier and
        # select GPC corner detection as the cognition method here
        # IMR_P7_T3_GPR_SETUP: Create the small particle filter, select
        # GPR weighting and define the GPR and observation uncertainties here
        #----------------------------------------------------------------

        self.datalog = DataLogger(log_dir="logs")
        # Wheels speeds in rad/s are encoded as a Vector3 with timestamp, 
        # with x for the right wheel and y for the left wheel.        
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",Vector3Stamped, self.true_wheel_speeds_callback,ip=self.robot_ip,
        )
        self.lidar_sub = Subscriber(
            "/lidar", RBLaserScan, self.lidar_callback, ip=self.robot_ip
        )
        self.groundtruth_sub = Subscriber(
            "/groundtruth", PoseStamped, self.groundtruth_callback, ip=self.robot_ip
        )

    def stopcommand(self):        
        print("Wheels stopping")
        r = Rate(10.0)

        self.stop_flag = True

        stop_msg = Vector3Stamped() # initially 0
        
        for i in range(10):
            stop_msg.vector.x=0
            stop_msg.vector.y=0
            self.wheel_speed_pub.publish(stop_msg)            
            r.sleep()

        self.lidar_sub.stop()
        self.true_wheel_speed_sub.stop()
        if self.simulation: self.groundtruth_sub.stop()

        print("Data saved in ",self.datalog.filename)
                 
    def true_wheel_speeds_callback(self, msg):
        print("Received sensed wheel speeds: R=", msg.vector.x,", L=", msg.vector.y)
        
        #-------------- Guided-practicals -------------------------------
        # IMR_RF2_T1_WHEEL_RATES: Store the measured right and left wheel
        # rates in the corresponding LaptopPilot attributes here
        #----------------------------------------------------------------
        # IMR_P2_T2_WHEEL_RATES: Store the measured right and left wheel
        # rates in the corresponding LaptopPilot attributes here
        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        #----------------------------------------------------------------
        
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received        
        print("Received lidar message", msg.header.seq)        
        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp()-msg.header.stamp
            self.sim_init = False     

        msg.header.stamp += self.sim_time_offset

        #-------------- Guided-practicals -------------------------------
        # IMR_RF2_T1_LIDAR_INTERPRET: Store and transform LiDAR measurements
        #----------------------------------------------------------------
        # IMR_P2_T2_LIDAR_DISPLAY: Store the LiDAR timestamp and raw
        # range-angle observations in the attributes used by show_laptop.py 
        self.lidar_timestamp_s = msg.header.stamp #we want the lidar measurement timestamp here
        
        self.lidar_data = np.zeros((len(msg.ranges), 2)) #specify length of the lidar data
        self.lidar_data[:,0] = msg.ranges # use ranges as a placeholder, workout northings in Task 4
        self.lidar_data[:,1] = msg.angles # use angles as a placeholder, workout eastings in Task 4  
        # IMR_P2_T4_LIDAR_TRANSFORM: Replace the raw display data with
        # LiDAR observations transformed into the Earth frame here.
         # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m #robot pose northings (see Task 3)
        p_eb[1] = self.est_pose_eastings_m #robot pose eastings (see Task 3)
        p_eb[2] = self.est_pose_yaw_rad #robot pose yaw (see Task 3)

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))        
                    
        z_lm = Vector(2)        
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]
                
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm) # see tutotial

            self.lidar_data[i,0] = t_em[0]
            self.lidar_data[i,1] = t_em[1]

        # this filters out any 
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
        #----------------------------------------------------------------
        # IMR_P7_T1_LIDAR_PROCESSING: Store the timestamp, transform valid
        # LiDAR observations for display, retain the range-bearing scan and
        # set the new-scan flag to True here
        #----------------------------------------------------------------

        self.datalog.log(msg, topic_name="/lidar")

    #-------------- Guided-practicals -------------------------------
    # IMR_P7_T2_DETECT_CORNER: Add the GPC corner-detection method here.
    # Classify the current scan, transform a detected corner into the
    # Earth frame and append it to self.corners for display.
    #----------------------------------------------------------------
    # IMR_P7_T3_GPR_WEIGHT: Add the GPR particle-weight update method
    # here using the current range-bearing scan.
    #----------------------------------------------------------------

    def groundtruth_callback(self, msg):
        """Log the ground-truth sample using the broadcast timestamp snapshot
        from the previous loop iteration, then buffer the current sample."""
        t = msg.header.stamp
        n = msg.pose.position.x
        e = msg.pose.position.y
        d = msg.pose.position.z
        q = [msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w]
        r = R.from_quat(q)  # [x, y, z, w]
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        yaw = np.mod(yaw, 360.0)

        # 1) If we have a previous sample buffered, log it now against the *previous* broadcast snapshot.
        if self.pending_groundtruth is not None:
            t_prev, n_prev, e_prev, d_prev, roll_prev, pitch_prev, yaw_prev = self.pending_groundtruth

            # Precise check: True if the previous ground-truth timestamp equals the broadcast timestamp
            is_broadcast_prev = (t_prev == self.broadcast)            

            with self.groundtruth_log.open('a') as f:
                f.write(
                    f"{t_prev},{t_prev - self.start_time},{n_prev},{e_prev},{d_prev},"
                    f"{roll_prev},{pitch_prev},{yaw_prev},{is_broadcast_prev}\n"
                )        
        self.pending_groundtruth = (t, n, e, d, roll, pitch, yaw)

    def pose_parse(self, msg, aruco = False):
        # parser converts pose data to a standard format for logging
        time_stamp = msg[0]
        

        if aruco == True:
            if self.sim_init == True: 
                self.sim_time_offset = datetime.utcnow().timestamp()-msg[0]               
                self.sim_init = False                                         
                
            # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse timeself.sim_time_offset
            print(
                "Received position update from",
                datetime.utcnow().timestamp() - msg[0] - self.sim_time_offset,
                "seconds ago",
            )            
            time_stamp = msg[0] + self.sim_time_offset

        pose_msg = PoseStamped() 
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0

        quat = Quaternion()        
        if self.simulation == False and aruco == True: quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else: quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat            
        
        return pose_msg

    #-------------- Guided-practicals -------------------------------
    # IMR_P3_T1_GENERATE_TRAJECTORY: Add a method here to:
    #   1. offset relative waypoints by the initial robot position;
    #   2. create a trajectory from the waypoint lists; and
    #   3. apply the chosen trajectory parameters.
    #----------------------------------------------------------------    
    def generate_trajectory(self):
        # pick waypoints as current pose relative or absolute northings and eastings
        if self.relative_path == True:
            for i in range(len(self.northings_path)):
                self.northings_path[i] += self.est_pose_northings_m #offset by current northings
                self.eastings_path[i] += self.est_pose_eastings_m #offset by current eastings

            # convert path to matrix and create a trajectory class instance
            C = l2m([self.northings_path, self.eastings_path])        
            self.path = TrajectoryGenerate(C[:, 0], C[:, 1])        
            
            # set trajectory variables (velocity, acceleration and turning arc radius)
            self.path.path_to_trajectory(self.v, self.a) #velocity and acceleration
            self.path.turning_arcs(self.arc_radius) #turning radius
            self.path.wp_id=0 #initialises the next waypoint

    def run(self, time_to_run=-1):
        self.start_time = datetime.utcnow().timestamp()
        
        try:
            r = Rate(10.0)
            while True:
                current_time = datetime.utcnow().timestamp()
                if time_to_run > 0 and current_time - self.start_time > time_to_run:
                    print("Time is up, stopping…")                    
                    break
                self.infinite_loop()
                r.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            self.stopcommand()


    def infinite_loop(self):
        """Main control loop

        Your code should go here.
        """
        ###############################################################
        # > Sense < #
        ###############################################################
        # get the latest position measurements
        aruco_pose = self.aruco_driver.read()    

        if aruco_pose is not None:             
            # converts aruco date to zeroros PoseStamped format                         
            msg = self.pose_parse(aruco_pose, aruco = True)
            self.broadcast = msg.header.stamp 

            # reads sensed pose for local use 
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()        
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2) # manage angle wrapping

            # logs the data            
            self.datalog.log(msg, topic_name="/aruco")

            #-------------- Guided-practicals -------------------------------
            # IMR_RF2_T1_INITIAL_POSE: On the first ArUco measurement, initialise
            # the estimated pose and motion-model timing here
            #----------------------------------------------------------------
            # IMR_P2_T3_INITIAL_POSE: On the first ArUco measurement, initialise
            # the estimated pose and motion-model timing here
            if self.initialise_pose == True:
                self.est_pose_northings_m = self.measured_pose_northings_m # use the aruco measured pose
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad
                

            # get current time and determine timestep
                self.t_prev = datetime.utcnow().timestamp() #initialise the time
                self.t = 0 #elapsed time
                time.sleep(0.1) #wait for approx a timestep before proceeding
                    
                    # path and tragectory are initialised
                self.initialise_pose = False 

            #----------------------------------------------------------------
            # IMR_P3_T1_INITIALISE_TRAJECTORY: Inside the pose-initialisation
            # conditional, generate the trajectory after setting the initial pose    
                self.generate_trajectory()
                print('check for  ini pose True')        
            #----------------------------------------------------------------

        ###############################################################
        # > Think < #
        ###############################################################

        #-------------- Guided-practicals -------------------------------
        # IMR_P7_T2_RUN_GPC: Run corner detection once when a new LiDAR
        # scan is available.
        #----------------------------------------------------------------        
        # IMR_P7_T3_RUN_GPR: Run GPR particle weighting once when a new
        # LiDAR scan is available.
        # IMR_P7_T3_MOVE_PARTICLES: In the GPR method After the specified elapsed time,
        # manually change two particle positions to investigate their weights.
        #----------------------------------------------------------------        
        
        ###############################################################
        #  Implement state estimation
        ###############################################################

        #-------------- Guided-practicals -------------------------------
        # IMR_RF2_T1_MOTION: Convert measured wheel rates to twist and
        # propagate the estimated pose over the current timestep
        #---------------------------------------------------------------- 
        # IMR_P2_T2_ESTIMATED_POSE: Change the estimated-pose attributes
        # below to verify that they are displayed and logged correctly.
        #self.est_pose_northings_m = self.measured_pose_northings_m # modify value
        #self.est_pose_eastings_m = self.measured_pose_eastings_m # modify value
        #self.est_pose_yaw_rad = self.measured_pose_yaw_rad # modify value
        #---------------------------------------------------------------- 
        # IMR_P2_T3_MOTION_MODEL: Once the pose is initialised, convert the
        # measured wheel rates to robot twist, calculate the timestep and
        # propagate the previous pose estimate here. Keep the estimate logging
        # and subsequent Act section inside the initialisation conditional.
        if self.initialise_pose != True:  
            print('check for  ini pose False') 
            q = Vector(2)            
            if self.measured_wheelrate_right is not None: q[0] = self.measured_wheelrate_right # wheel rate rad/s (measured)
            if self.measured_wheelrate_left is not None: q[1] = self.measured_wheelrate_left # wheel rate rad/s (measured)
            u = self.ddrive.fwd_kinematics(q)
            print('twist',u)            
        
            #determine the time step
            t_now = datetime.utcnow().timestamp()        
                
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop

            # take current pose estimate and update by twist
            p_robot = Vector(3)
            p_robot[0] = self.est_pose_northings_m
            p_robot[1] = self.est_pose_eastings_m
            p_robot[2] = self.est_pose_yaw_rad
                                
            p_robot = rigid_body_kinematics(p_robot, u, dt)
            p_robot[2] = p_robot[2] % (2 * np.pi)  # deal with angle wrapping          

            # update for show_laptop.py            
            self.est_pose_northings_m = self.measured_pose_northings_m
            self.est_pose_eastings_m = self.measured_pose_eastings_m
            self.est_pose_yaw_rad = self.measured_pose_yaw_rad
        ################### Motion Model ##############################
        # convert true wheel speeds in to twist
        
        #----------------------------------------------------------------

            msg = self.pose_parse([datetime.utcnow().timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad])
            self.datalog.log(msg, topic_name="/est_pose")

        #-------------- Guided-practicals -------------------------------
        # IMR_RF2_T2_FREESPACE_CALC: Determine freespace direction
        #----------------------------------------------------------------


        #################################################################
        #  Implement control 
        #################################################################

        #-------------- Guided-practicals -------------------------------
        # IMR_P3_T1_SAMPLE_TRAJECTORY: Update waypoint progress and sample
        # the reference pose and feedforward twist at the current elapsed time
        #################### Trajectory sample #################################    

        # feedforward control: check wp progress and sample reference trajectory
        #print("path ",self.path)
        #self.path.wp_progress(self.t, p_robot, self.arc_radius) # fill turning radius
        #p_ref, u_ref = self.path.p_u_sample(self.t) #sample the path at the current elapsetime (i.e., seconds from start of motion modelling)

        if self.path is not None:
            self.path.wp_progress(self.t, p_robot, self.arc_radius)
            p_ref, u_ref = self.path.p_u_sample(self.t)
            print("p-ref[0]",p_ref[0])
            #print('pos', self.est_pose_northings_m)
            #self.est_pose_northings_m = p_ref[0][0]
            #self.est_pose_eastings_m = p_ref[1][0]
            #self.est_pose_yaw_rad = p_ref[2,0]


        # IMR_P3_T2_POSE_ERROR: Calculate the difference between the
        # reference and estimated poses, wrap the yaw error and express the
        # pose error in the robot body frame.
        #<sample path at current elapse time>

        # feedback control: get pose change to desired trajectory from body
            dp = p_ref - p_robot #compute difference between reference and estimated pose in the $e$-frame
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw
            H_eb = HomogeneousTransformation(p_robot[0:2], p_robot[2])
            ds =  Inverse(H_eb.H_R)@dp # rotate the $e$-frame difference to get it in the $b$-frame (Hint: dp_b = H_be.H_R @ dp_e)
            # IMR_P3_T2_FEEDBACK_CONTROL: Initialise or update the control
            # gains, calculate the feedback correction and combine it with the
            # feedforward twist.
            
            # compute control gains for the initial condition (where the robot is stationalry)
            self.k_s = 1/self.tau_s #ks
            if self.initialise_control == True:
                self.k_n = (2*u_ref[0])/(self.L**2) #kn
                self.k_g = u_ref[0]/self.L #kg
                self.initialise_control = False # maths changes a bit after the first iteration
            # update the controls
            du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

            # total control
            u = u_ref + du # combine feedback and feedforward control twist components

            # update control gains for the next timestep
            self.k_n = (2*u[0])/(self.L**2) #kn
            self.k_g = u[0]/self.L #kg
            #----------------------------------------------------------------

            #-------------- Guided-practicals -------------------------------
            # IMR_RF2_T1_WHEEL_COMMANDS: Change the right and left wheel rates
            # here to produce twist, linear and rotational motion.
            # IMR_RF2_T2_FREESPACE_STEER: Steer towards freespace direction
            # IMR_RF2_T2_TWIST_CMD: Convert modified twist to wheel rates
            # IMR_RF2_T3_BRAKING: Brake when obstacles approach
            # IMR_RF2_T3_TWIST_CMD: Convert modified twist to wheel rates
            # IMR_RF2_T4_ESCAPE: Check if trapped and escape
            #----------------------------------------------------------------
            # IMR_P3_T2_ACTUATOR_COMMANDS: Limit the commanded linear and
            # angular velocities, convert the resulting twist to wheel rates
            # and store the right and left wheel commands below
            # ensure within performance limitation
            if u[0] > self.v_max: u[0] = self.v_max
            if u[0] < -self.v_max: u[0] = -self.v_max
            if u[1] > self.w_max: u[1] = self.w_max
            if u[1] < -self.w_max: u[1] = -self.w_max

            # actuator commands                 
            q = self.ddrive.inv_kinematics(u)            

            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0,0] # Right wheelspeed rad/s
            wheel_speed_msg.vector.y = q[1,0] # Left wheelspeed rad/s
        #----------------------------------------------------------------
        # IMR_P2_T1_WHEEL_COMMANDS: Change the right and left wheel rates
        # here to produce twist, linear and rotational motion.
        #----------------------------------------------------------------              
        # IMR_P7_T1_STATIONARY: Set both wheel-rate commands to zero for
        # the stationary cognition experiments.  
        #----------------------------------------------------------------      

        wheel_speed_msg = Vector3Stamped()
        #wheel_speed_msg.vector.x = 1 * np.pi  # Right wheel 0.5 rev/s = 1*pi rad/s
        #wheel_speed_msg.vector.y = 2 * np.pi  # Left wheel 1 rev/s = 2*pi rad/s


        self.cmd_wheelrate_right = wheel_speed_msg.vector.x
        self.cmd_wheelrate_left = wheel_speed_msg.vector.y

        ################################################################################
        # > Act < #
        ################################################################################        
        # Send commands to the robot        
        if self.stop_flag == False: self.wheel_speed_pub.publish(wheel_speed_msg)
        self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    args = parser.parse_args()

    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)
