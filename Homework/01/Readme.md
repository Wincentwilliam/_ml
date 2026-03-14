# Traveling Salesperson Problem (TSP) - Hill Climbing

This project implements the Hill Climbing optimization algorithm to solve the Traveling Salesperson Problem (TSP). The goal is to find the shortest possible route that visits a set of cities and returns to the starting point.

## Implementation Details

| Component | Explanation |
| :--- | :--- |
| **State** | Represents the sequence of cities visited (a permutation of indices). |
| **Neighbor** | Uses the **2-opt strategy**: two edges are selected and swapped to uncross paths, effectively reversing a segment of the route. |
| **Height** | Calculated as `-1 * total_distance`. Since Hill Climbing aims to *maximize* a value, we negate the distance to turn the minimization problem into a maximization problem. |
| **Deterministic** | `random.seed(0)` is used to ensure the algorithm produces the exact same result every time it is executed, aiding in debugging and testing. |
| **Hill Climbing** | A local search algorithm that iteratively modifies the current solution by generating a neighbor and accepting it if it improves the "height" (distance). |

## How to Run
1. Ensure you have Python installed.
2. Open your terminal in the project directory.
3. Activate your virtual environment (if applicable):
   ```bash
   .\.venv\Scripts\Activate.ps1

---

### Important Reminder for `HillClimbing.py`
To ensure the `README` documentation matches your code behavior, make sure your `HillClimbing.py` starts exactly like this:

```python
import math
import random

# Setting the seed ensures the result is the same every time you run it
random.seed(0) 

# ... the rest of your code ...