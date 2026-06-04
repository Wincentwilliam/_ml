import random
from typing import List, Tuple

def get_quiz_data() -> List[Tuple[str, str]]:
    """Returns a list of (Country, Capital) tuples."""
    return [
        ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
        ("Canada", "Ottawa"), ("Brazil", "Brasilia"), ("Australia", "Canberra"),
        ("India", "New Delhi"), ("United Kingdom", "London"), ("Italy", "Rome"),
        ("Spain", "Madrid"), ("Mexico", "Mexico City"), ("China", "Beijing"),
        ("South Korea", "Seoul"), ("Egypt", "Cairo"), ("Argentina", "Buenos Aires"),
        ("Russia", "Moscow"), ("Thailand", "Bangkok"), ("Turkey", "Ankara"),
        ("Kenya", "Nairobi"), ("Norway", "Oslo")
    ]

def run_quiz():
    """Executes the quiz logic without requiring user input via a simulated sequence."""
    questions = get_quiz_data()
    random.shuffle(questions)
    
    # Since the prompt asks for an interactive game but the system rules 
    # forbid input(), we simulate a user's attempt to demonstrate functionality.
    # In a real scenario, this would be replaced by input() calls.
    simulated_answers = [
        "Paris", "Berlin", "Tokyo", "Toronto", "Brasilia", 
        "Sydney", "New Delhi", "London", "Rome", "Madrid", 
        "Mexico City", "Beijing", "Seoul", "Cairo", "Buenos Aires", 
        "Moscow", "Bangkok", "Ankara", "Nairobi", "Oslo"
    ]
    
    score = 0
    print("--- World Capitals Quiz ---")
    print("Simulating user responses as per system constraints...\n")

    for i in range(len(questions)):
        country, correct_capital = questions[i]
        # Simulate picking an answer from the simulated list
        user_answer = simulated_answers[i % len(simulated_answers)]
        
        print(f"Question {i+1}: What is the capital of {country}?")
        print(f"User Answer: {user_answer}")
        
        if user_answer.lower() == correct_capital.lower():
            print("Result: Correct!")
            score += 1
        else:
            print(f"Result: Wrong! The correct answer is {correct_capital}.")
        print("-" * 30)

    print(f"\nQuiz Complete!")
    print(f"Final Score: {score}/{len(questions)}")
    percentage = (score / len(questions)) * 100
    print(f"Grade: {percentage}%")

if __name__ == '__main__':
    run_quiz()