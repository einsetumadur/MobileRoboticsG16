import cv2
import tkinter as tk
from PIL import Image, ImageTk
import vision
import numpy as np

unwarp = True

minconv = 0.3
robpos = np.array([-100,-100])
ori = np.pi/2
pxpcm = 5

# Function to update the image displayed from the webcam
def update_frame():
    global robpos
    global ori
    global pxpcm
    ret, frame = cap.read()
    if ret:
        if unwarp:
            frame = cv2.warpPerspective(frame,Tmap,(1000,700))
        HLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS_FULL)
        compd = vision.visualizer(HLS)
        mixed = cv2.addWeighted(frame,0.5,compd,0.5,0)
        found,center,scl,orient = vision.get_Robot_position_orientation(HLS,(100,255,4),5)
    
        if found:
            robpos = center
            ori = orient
            pxpcm = scl
            mixed = vision.paint_robot(mixed,(0,0,255),robpos,ori,pxpcm)
        else: 
            mixed = vision.paint_robot(mixed,(0,0,100),robpos,ori,pxpcm)

        frame = cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        panel.img = img
        panel.config(image=img)
        panel.after(10, update_frame)

def set_minconv(val):
    global minconv
    minconv = int(val)/1000
    print(minconv)

# Function for button click
def on_button_click():
    print("hey !")
    pass  # Add your button logic here

# Initialize OpenCV VideoCapture object
cap = cv2.VideoCapture(0)

if unwarp:
    Tmap = vision.get_warp(cap,(1000,700),20,10)

# Create Tkinter window
root = tk.Tk()
root.title("Mobile Robotics")

# Create left frame for buttons and trackbars
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create trackbars
scale = tk.Scale(left_frame, from_=0, to=1000, orient=tk.HORIZONTAL, label=f"minConv",command=lambda val: set_minconv(val))
scale.pack()

# Create buttons
button = tk.Button(left_frame, text=f"By Color", command=lambda i=0: on_button_click())
button.pack(pady=5)

# Create right frame for displaying webcam feed
right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Create panel to display webcam feed
panel = tk.Label(right_frame)
panel.pack()

# Start updating the webcam feed
update_frame()

root.mainloop()

# Release the VideoCapture and destroy all OpenCV windows when done
cap.release()
cv2.destroyAllWindows()