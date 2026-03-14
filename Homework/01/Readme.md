# Traveling Salesperson Problem (TSP) - Hill Climbing Solver

This project implements a **Hill Climbing algorithm** to solve the Traveling Salesperson Problem. The objective is to find the shortest possible route (a Hamiltonian cycle) that visits a set of given cities exactly once.

## Project Overview

| Component | Description |
| :--- | :--- |
| **Algorithm** | Hill Climbing (Local Search) |
| **Optimization Goal** | Minimize total distance (by maximizing negative distance) |
| **Neighbor Strategy** | 2-opt Swap (reversing a segment of the path to uncross lines) |
| **Reproducibility** | Deterministic (using `random.seed(0)`) |
| **State** | A permutation of city indices `[0, 1, 2, ..., n]` |

## Core Logic Implementation

The solution relies on the following core methods:

### 1. Height Function
Since Hill Climbing naturally seeks to reach the highest point, we define "height" as the negative value of the total distance.
```python
def height(self) -> float:
    # Maximizing negative distance is equivalent to minimizing distance.
    return -self.total_distance()
```
### 2. Neighbor Function (2-opt)
```python
def neighbor(self) -> 'TSPSolution':
    neighbor_sol = TSPSolution(self.cities)
    neighbor_sol.path = self.path[:]
    # Select two random indices and reverse the segment between them
    i, j = sorted(random.sample(range(len(self.path)), 2))
    neighbor_sol.path[i:j+1] = reversed(neighbor_sol.path[i:j+1])
    return neighbor_sol
```
### 3. Hill Climbing Execution
```python
def hill_climbing(initial_solution: TSPSolution):
    current = initial_solution
    while True:
        neighbor = current.neighbor()
        # Move to neighbor only if the distance is shorter (height is higher)
        if neighbor.height() > current.height():
            current = neighbor
        else:
            break # Local optimum reached
    return current
```

## How to Run
1. Ensure Python 3.x is installed.
2. Open your project folder in the terminal.
3. Activate the virtual environment:
```bash
.\.venv\Scripts\Activate.ps1
```
4. Run the main script:
```bash
python HillClimbing.py
```

## AI Conversation Record
1. Problem Definition: Implemented TSP using Hill Climbing with strict constraints on neighbor and height methods.
2. Logic Refinement: Utilized 2-opt for path optimization and negative distance for the height function.
3. Deterministic Results: Applied random.seed(0) to ensure identical outputs during testing and debugging.
4. Professional Standards: Added type hinting and docstrings for better code clarity and maintainability.

---

### Instructions for you:
1. **Create/Open** the `README.md` file in your VS Code.
2. **Delete everything** currently in that file.
3. **Paste** the code block above into the file.
4. **Save** the file. 