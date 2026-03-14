import math
import random
from typing import List, Tuple

# Fix the seed for deterministic, repeatable results
random.seed(0)

class TSPSolution:
    """Represents a route for the Traveling Salesperson Problem."""
    
    def __init__(self, cities: List[Tuple[int, int]]):
        self.cities = cities
        self.path = list(range(len(cities)))
        
    def distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculates Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def total_distance(self) -> float:
        """Calculates the sum of distances of the entire path."""
        dist = 0.0
        for i in range(len(self.path)):
            c1 = self.cities[self.path[i]]
            c2 = self.cities[self.path[(i + 1) % len(self.path)]]
            dist += self.distance(c1, c2)
        return dist

    def height(self) -> float:
        """The 'Height' for Hill Climbing is the negative total distance."""
        return -self.total_distance()

    def neighbor(self) -> 'TSPSolution':
        """Generates a neighbor using the 2-opt swap (reversing a segment)."""
        neighbor_sol = TSPSolution(self.cities)
        neighbor_sol.path = self.path[:]
        
        # Select two random points and reverse the path between them
        i, j = sorted(random.sample(range(len(self.path)), 2))
        neighbor_sol.path[i:j+1] = reversed(neighbor_sol.path[i:j+1])
        return neighbor_sol

def hill_climbing(initial_solution: TSPSolution, iterations: int = 1000):
    """Performs the Hill Climbing algorithm."""
    current = initial_solution
    
    for i in range(iterations):
        neighbor = current.neighbor()
        # If the neighbor is 'higher' (shorter distance), move there
        if neighbor.height() > current.height():
            print(f"Iteration {i}: Found better distance { -neighbor.height():.2f}")
            current = neighbor
        else:
            continue # Keep looking for improvements
            
    return current

if __name__ == "__main__":
    # Test Data: 10 Cities
    cities = [(208, 84), (75, 45), (190, 120), (25, 110), (140, 10), 
              (80, 200), (150, 150), (200, 200), (50, 50), (100, 120)]
    
    print("--- Starting Hill Climbing ---")
    initial_sol = TSPSolution(cities)
    print(f"Initial Distance: {initial_sol.total_distance():.2f}")
    
    best = hill_climbing(initial_sol)
    
    print("\n--- Final Results ---")
    print(f"Best Path: {best.path}")
    print(f"Best Distance: { -best.height():.2f}")