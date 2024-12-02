# Description: This is the main script for the face tracking robot. It uses the URBasic library to control the robot and OpenCV to detect faces and objects in the video stream.
import URBasic
import math
import time
import cv2
from imutils.video import VideoStream
from detection import find_faces_dnn, detect_objects_yolo, show_frame
from detection import vs  # Ensure vs is properly initialized in detection module
from robot_control import set_lookorigin, move_to_face 

"""SETTINGS AND VARIABLES ________________________________________________________________"""
ROBOT_IP = '192.168.56.101'
ACCELERATION = 0.4  # Robot acceleration value
VELOCITY = 0.4  # Robot speed value

# The Joint position the robot starts at
robot_startposition = (math.radians(0),
                    math.radians(-78),
                    math.radians(-93),
                    math.radians(-15),
                    math.radians(90),
                    math.radians(0))

# Initialize robot and robotModel
ROBOT_IP = '192.168.56.101'
robotModel = URBasic.robotModel.RobotModel()
robot = URBasic.urScriptExt.UrScriptExt(host=ROBOT_IP, robotModel=robotModel)


"""FACE TRACKING LOOP ____________________________________________________________________"""

# initialise robot with URBasic
print("initialising robot")

robot.reset_error()
print("robot initialised")
time.sleep(1)

# Move Robot to the midpoint of the lookplane
robot.movej(q=robot_startposition, a= ACCELERATION, v= VELOCITY )

robot_position = [0,0]
origin = set_lookorigin(robot)  # Set the origin of the robot coordinate system
robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
time.sleep(1) # just a short wait to make sure everything is initialised

print("Choose tracking mode: 'face' or 'object'")
mode = input().strip().lower()
try:
    print("starting loop")
    while True:
        frame = vs.read()
        if frame is None:
            print("No frame captured from video stream")
            continue
        
        if mode == 'face':
            target_positions, new_frame = find_faces_dnn(frame)
        elif mode == 'object':
            target_positions, labels, new_frame = detect_objects_yolo(frame)
            # Example: Track the first "chair"
            target_positions = [pos for pos, label in zip(target_positions, labels) if label == "cell phone"]
        
        show_frame(new_frame)
        if target_positions:
            robot_position = move_to_face(target_positions, robot_position, robot, origin)
        if cv2.waitKey(1) & 0xFF == ord('q'):    
            print("exiting loop")
            robot.close() # Close the robot connection
            vs.stop()  # Stop the VideoStream
            cv2.destroyAllWindows()  # Close all OpenCV windows
            break
except KeyboardInterrupt:
    print("closing robot connection")
    # Remember to always close the robot connection, otherwise it is not possible to reconnect
    robot.close()
    vs.stop()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
    robot.close()
    vs.stop()
    cv2.destroyAllWindows()