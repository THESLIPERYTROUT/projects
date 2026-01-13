import numpy as np
import matplotlib.pyplot as plt

mass: float = 5.0  # kg
stiffness: float = 2000.0  # N/m
damping: float = 10.0  # Ns/m
damping_ratio: float = damping / (2 * np.sqrt(mass * stiffness))  # Corrected

def underdamped_solution(mass, stiffness, damping, R, phi):
    system_type = "Underdamped"
    t = np.linspace(0, 10, 1000)
    wn = np.sqrt(stiffness / mass)
    zeta = damping / (2 * mass * wn)
    wd = wn * np.sqrt(1 - zeta**2)
    solution = R * np.exp(-zeta * wn * t) * np.cos(wd * t - phi)
    
    plt.plot(t, solution)
    plt.title(f'{system_type} Vibration Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.grid(True)
    plt.savefig('underdamped_response.png', dpi=150, bbox_inches='tight')
    plt.show()

def critically_damped_solution(mass, stiffness, damping, A, B):
    system_type = "Critically Damped"   
    t = np.linspace(0, 10, 1000)
    wn = np.sqrt(stiffness / mass)
    solution = (A + B * t) * np.exp(-wn * t)
    
    plt.plot(t, solution)
    plt.title(f'{system_type} Vibration Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.grid(True)
    plt.savefig('critically_damped_response.png', dpi=150, bbox_inches='tight')
    plt.show()

def overdamped_solution(mass, stiffness, damping, A, B):
    system_type = "Overdamped"
    t = np.linspace(0, 10, 1000)
    discr = damping**2 - 4 * mass * stiffness
    if discr <= 0:
        raise ValueError("Discriminant is not positive; system is not overdamped.")
    r1 = (-damping + np.sqrt(discr)) / (2 * mass)
    r2 = (-damping - np.sqrt(discr)) / (2 * mass)
    solution = A * np.exp(r1 * t) + B * np.exp(r2 * t)
    
    plt.plot(t, solution)
    plt.title(f'{system_type} Vibration Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement')
    plt.grid(True)
    plt.savefig('overdamped_response.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    if damping_ratio < 1:
        underdamped_solution(mass, stiffness, damping, R=1.0, phi=0.0)
    elif damping_ratio == 1:
        critically_damped_solution(mass, stiffness, damping, A=1.0, B=0.0)
    else:
        overdamped_solution(mass, stiffness, damping, A=1.0, B=0.0)

if __name__ == "__main__":
    main()


