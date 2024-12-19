import numpy as np

def calculate_sound_direction():
    
    SPEED_OF_SOUND = 343  # m/s
    DISTANCE = 0.0226     # 22.6 cm in meters

    try:
        t1 = float(input("Enter t1 time (in seconds): "))
        t3 = float(input("Enter t3 time (in seconds): "))
        
        time_diff = t3 - t1
        ratio = (time_diff * SPEED_OF_SOUND) / DISTANCE
        ratio = np.clip(ratio, -1, 1)  
        theta = np.pi/2 - np.arccos(ratio)  # θ = π/2 - cos⁻¹((t3-t1)*343/0.0226) 
        theta_degrees = np.degrees(theta)

        print(f"\nCalculated Direction:")
        print(f"Angle θ = {theta_degrees:.2f} degrees")

    except ValueError:
        print("Please enter valid numerical values for times")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    print("Sound Direction Calculator")
    print("-------------------------")
    calculate_sound_direction()
    