import math
import random

# Add this line to make the results the same every time you run it
random.seed(0) 

class TSPSolution:
    def __init__(self, cities):
        self.cities = cities
        # Initial solution: 0 -> 1 -> 2 -> ... -> n
        self.path = list(range(len(cities)))
        
    def distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def total_distance(self):
        dist = 0
        for i in range(len(self.path)):
            c1 = self.cities[self.path[i]]
            c2 = self.cities[self.path[(i + 1) % len(self.path)]]
            dist += self.distance(c1, c2)
        return dist

    def height(self):
        return -self.total_distance()

    def neighbor(self):
        neighbor_sol = TSPSolution(self.cities)
        neighbor_sol.path = self.path[:]
        
        # Now these random samples will follow a predictable sequence
        i, j = sorted(random.sample(range(len(self.path)), 2))
        
        neighbor_sol.path[i:j+1] = reversed(neighbor_sol.path[i:j+1])
        return neighbor_sol

def hill_climbing(initial_solution):
    current = initial_solution
    while True:
        neighbor = current.neighbor()
        if neighbor.height() > current.height():
            current = neighbor
        else:
            break
    return current

# --- Execution ---
cities = [(208, 84), (75, 45), (190, 120), (25, 110), (140, 10), 
          (80, 200), (150, 150), (200, 200), (50, 50), (100, 120)]

sol = TSPSolution(cities)
best = hill_climbing(sol)
print("Best Path:", best.path)
print("Min Distance:", -best.height())