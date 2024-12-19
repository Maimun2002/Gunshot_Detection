import numpy as np
from scipy.optimize import minimize

SPEED_OF_SOUND = 343  # m/s

def get_input_matrix(mic_count):
    receivers = []
    print(f"Enter the positions of {mic_count} microphones (as x, y coordinates):")
    for i in range(mic_count):
        x = float(input(f"Enter x-coordinate of microphone {i+1}: "))
        y = float(input(f"Enter y-coordinate of microphone {i+1}: "))
        receivers.append([x, y])
    return np.array(receivers)

def get_tdoa_values(mic_count):
    TDoA = []
    print(f"\nEnter the Time Difference of Arrival (TDoA) between mic 1 and the other {mic_count - 1} mics:")
    for i in range(1, mic_count):
        tdoa = float(input(f"TDoA between mic 1 and mic {i + 1} (in seconds): "))
        TDoA.append(tdoa)
    return np.array(TDoA)

def tdoa_error(source, receivers, TDoA, speed_of_sound):
    errors = []
    for i, tdoa in enumerate(TDoA):
        dist_1 = np.linalg.norm(source - receivers[0])  # Distance from source to mic 1
        dist_i = np.linalg.norm(source - receivers[i + 1])  # Distance from source to other mics

        expected_diff = tdoa * speed_of_sound  
        errors.append((dist_1 - dist_i - expected_diff)**2)  

    return np.sum(errors)  

def main():
    mic_count = int(input("Enter the number of mics: "))  
    receivers = get_input_matrix(mic_count)
    TDoA = get_tdoa_values(mic_count)

    initial_guess = np.array([5, 5])  
    
    result = minimize(
        tdoa_error,
        initial_guess,
        args=(receivers, TDoA, SPEED_OF_SOUND),
        method='Nelder-Mead'
    )

    source_position = result.x

    print(f"\nEstimated source position: {source_position}")

if __name__ == "__main__":
    print("Sound Source Localization Using TDoA")
    print("-------------------------------------")
    main()
