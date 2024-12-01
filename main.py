import numpy as np
import cv2
import pandas as pd

# Load city coordinates from a CSV file and scale them for visualization
def load_cities_from_csv(file_path, width, height):
    data = pd.read_csv(file_path)
    cities = []
    lat_min, lat_max = 35, 42  # Latitude boundaries for Turkey
    lon_min, lon_max = 25, 45  # Longitude boundaries for Turkey

    # Iterate over the rows of the CSV and scale latitude/longitude to fit the visualization dimensions
    for _, row in data.iterrows():
        x = int((row["longitude"] - lon_min) / (lon_max - lon_min) * width)
        y = int((lat_max - row["latitude"]) / (lat_max - lat_min) * height)
        cities.append((x, y))
    return cities, data["city"].tolist()

# Generate a random initial solution (random permutation of city indices)
def initialize_solution(count):
    solution = np.arange(count)  # Create an array of city indices
    np.random.shuffle(solution)  # Shuffle the array to randomize the order
    return solution

# Evaluate the total distance of a solution (path connecting cities in a given order)
def evaluate_solution(cities, solution):
    total_distance = 0
    for i in range(len(solution)):
        a = solution[i]
        b = solution[i-1]
        dx = cities[a][0] - cities[b][0]
        dy = cities[a][1] - cities[b][1]
        total_distance += (dx**2 + dy**2)**0.5  # Add Euclidean distance between two cities
    return total_distance

# Create a new solution by swapping two cities in the current solution
def modify_solution(current):
    new_solution = current.copy()
    a, b = np.random.choice(len(current), size=2, replace=False)  # Pick two random indices
    new_solution[a], new_solution[b] = new_solution[b], new_solution[a]  # Swap cities
    return new_solution

# Visualize the current solution by drawing the cities and connecting lines
def draw_solution(width, height, cities, solution, infos, city_names):
    frame = np.zeros((height, width, 3), dtype=np.uint8)  # Create a blank image
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5

    # Draw lines connecting the cities in the current order
    for i in range(len(solution)):
        a = solution[i]
        b = solution[i-1]
        cv2.line(frame, cities[a], cities[b], green, 1)

    # Draw cities as circles and label them with city names
    for i, city in enumerate(cities):
        cv2.circle(frame, city, 5, red, -1)
        cv2.putText(frame, city_names[i], (city[0]+5, city[1]-5), font, size, white, 1)

    # Display additional information (temperature, scores) on the screen
    temperature, current_score, best_score, worst_score = infos
    cv2.putText(frame, f"Temperature: {temperature:.2f}", (10, 20), font, size, white, 1)
    cv2.putText(frame, f"Current: {current_score:.2f}", (10, 40), font, size, white, 1)
    cv2.putText(frame, f"Best: {best_score:.2f}", (10, 60), font, size, white, 1)
    cv2.putText(frame, f"Worst: {worst_score:.2f}", (10, 80), font, size, white, 1)

    # Show the image in a window
    cv2.imshow("Simulated Annealing", frame)
    cv2.waitKey(3)

# Main program begin
if __name__ == "__main__":

    # Parameters for visualization and simulated annealing
    width, height = 1280, 720
    initial_temperature = 50000
    stopping_temperature = 0.1
    temperature_decay = 0.999

    # Load city data from a CSV file
    csv_path = "C:\\Users\\betul\\OneDrive\\Masaüstü\\SimulatedAnnealing\\citiesofturkey.csv"
    cities, city_names = load_cities_from_csv(csv_path, width, height)
    city_count = len(cities)

    # Initialize the simulated annealing process
    current_solution = initialize_solution(city_count)
    current_score = evaluate_solution(cities, current_solution)

    best_score = current_score
    worst_score = current_score

    temperature = initial_temperature

    # Simulated annealing loop
    while temperature > stopping_temperature:
        # Generate a new solution by modifying the current one
        new_solution = modify_solution(current_solution)
        new_score = evaluate_solution(cities, new_solution)

        # Update the best and worst scores
        best_score = min(best_score, new_score)
        worst_score = max(worst_score, new_score)

        # Accept the new solution if it's better, or probabilistically if it's worse
        if new_score < current_score:
            current_solution = new_solution
            current_score = new_score
        else:
            delta = new_score - current_score
            probability = np.exp(-delta / temperature)
            if np.random.uniform() < probability:
                current_solution = new_solution
                current_score = new_score

        # Reduce the temperature (cooling schedule)
        temperature *= temperature_decay

        # Visualize the current solution
        draw_solution(width, height, cities, current_solution, 
                     (temperature, current_score, best_score, worst_score),
                     city_names)
