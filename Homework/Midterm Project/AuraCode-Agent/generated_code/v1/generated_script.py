def create_multiplication_table(size=10):
    """Create and display a formatted multiplication table."""
    # Create the header row
    header = "   " + "  ".join(f"{i:2}" for i in range(1, size + 1))
    print(header)
    
    # Create separator line
    separator = "-" * len(header)
    print(separator)
    
    # Create each row
    for i in range(1, size + 1):
        row = f"{i:2} | " + "  ".join(f"{i * j:2}" for j in range(1, size + 1))
        print(row)

if __name__ == '__main__':
    create_multiplication_table(10)