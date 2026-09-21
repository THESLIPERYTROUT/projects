import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_projectile_trajectory(initial_velocity=None, launch_angle=None, gravity=9.81, time_to_tar=None, sdistance_x=None, sdistance_y=None, fdistance_x=None, fdistance_y=None):
    if initial_velocity == None: 
        try:    
            