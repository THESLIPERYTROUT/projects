import math
import matplotlib.pyplot as plt

# all lengths in inches
lid_weight: int = 40 #pounds
lid_cm_x: float = 14.622 #cm="center of mass"
lid_cm_y: float = -7.887

gas_strut_compression_force: int = 45 #lbf
gas_strut_extension_force: int = 40
gas_strut_stroke: float = 9.84 
gas_strut_force_per_length: float = (gas_strut_extension_force-gas_strut_compression_force)/gas_strut_stroke
gas_strut_base_to_axis: float = 20.79 
gas_strut_top_to_axis: float = 3.575 


def lid_0_to_90_degs(lid_weight, lid_cm_x, lid_cm_y, gas_strut_force_per_length, gas_strut_base_to_axis, gas_strut_top_to_axis):
    domain = 90 #degs
    for degree in range(domain + 1):
        rad_weight = math.radians(degree)
        rad_strut = math.radians(degree)
        weight_torque_value = weight_torque(rad_weight, lid_weight, lid_cm_x, lid_cm_y)
        strut_torque_value = strut_torque(rad_strut, gas_strut_force_per_length, gas_strut_base_to_axis, gas_strut_top_to_axis)
        net_torque = strut_torque_value - weight_torque_value
        plt.scatter(degree, net_torque, color='blue', s=20)
        plt.scatter(degree, weight_torque_value, color='red', s=20)
        plt.scatter(degree, strut_torque_value, color='green', s=20)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Lid Angle (degrees)')
    plt.ylabel('Torque (lb-in)')
    plt.title('Net Torque vs Lid Angle')
    plt.legend(['Net Torque', 'Weight Torque', 'Strut Torque'], loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig('lid_torque.png', dpi=150, bbox_inches='tight')
        
def weight_torque(rad_weight, lid_weight, lid_cm_x, lid_cm_y): #cm="center of mass"
    rad = rad_weight
    cm_x_pos = lid_cm_x * math.cos(rad) - lid_cm_y * math.sin(rad) #rotation matrix application
    weight_torque  = lid_weight * cm_x_pos
    return weight_torque

def strut_torque(rad_strut, gas_strut_force_per_length, gas_strut_base_to_axis, gas_strut_top_to_axis):
    rad = rad_strut
    strut_length = math.sqrt(gas_strut_base_to_axis**2 + gas_strut_top_to_axis**2 - 2 * gas_strut_base_to_axis * gas_strut_top_to_axis * math.cos(rad)) #law of cosines
    strut_angle = math.acos((gas_strut_base_to_axis**2 - strut_length**2 - gas_strut_top_to_axis**2) / (-2 * gas_strut_base_to_axis * strut_length)) #law of cosines rearranged
    strut_torque = 2*gas_strut_force_per_length * math.sin(strut_angle) * gas_strut_top_to_axis
    return strut_torque

def main():
    lid_0_to_90_degs(lid_weight, lid_cm_x, lid_cm_y, gas_strut_force_per_length, gas_strut_base_to_axis, gas_strut_top_to_axis)
    print(1)

if __name__ == "__main__":
    main()