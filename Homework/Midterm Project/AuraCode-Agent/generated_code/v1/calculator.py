def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        return "Error: Division by zero"
    return x / y

def display_menu():
    print("\n=== Simple Calculator ===")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Exit")
    print("=========================")

def get_numbers():
    try:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        return num1, num2
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None, None

def main():
    while True:
        display_menu()
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            numbers = get_numbers()
            if numbers:
                num1, num2 = numbers
                result = add(num1, num2)
                print(f"Result: {num1} + {num2} = {result}")
        
        elif choice == '2':
            numbers = get_numbers()
            if numbers:
                num1, num2 = numbers
                result = subtract(num1, num2)
                print(f"Result: {num1} - {num2} = {result}")
        
        elif choice == '3':
            numbers = get_numbers()
            if numbers:
                num1, num2 = numbers
                result = multiply(num1, num2)
                print(f"Result: {num1} * {num2} = {result}")
        
        elif choice == '4':
            numbers = get_numbers()
            if numbers:
                num1, num2 = numbers
                result = divide(num1, num2)
                if isinstance(result, str):
                    print(result)
                else:
                    print(f"Result: {num1} / {num2} = {result}")
        
        elif choice == '5':
            print("Exiting calculator. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please select 1-5.")
        
        print()

if __name__ == '__main__':
    main()