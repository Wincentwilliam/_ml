#include <iostream>
#include <vector>
#include <string>

struct TestCase {
    double n1;
    double n2;
    char op;
};

int main() {
    // Hardcoded test cases to replace interactive input and prevent TimeoutError
    std::vector<TestCase> tests = {
        {10.5, 5.5, '+'},
        {20.0, 4.0, '-'},
        {6.0, 7.0, '*'},
        {100.0, 5.0, '/'},
        {10.0, 0.0, '/'}, // Test division by zero
        {10.0, 2.0, '?'}  // Test invalid operator
    };

    std::cout << "Simple C++ Calculator (Automated Demo)" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    for (const auto& test : tests) {
        double num1 = test.n1;
        double num2 = test.n2;
        char op = test.op;

        std::cout << "Input: " << num1 << " " << op << " " << num2 << " -> ";
        
        switch (op) {
            case '+':
                std::cout << "Result: " << num1 + num2 << std::endl;
                break;
            case '-':
                std::cout << "Result: " << num1 - num2 << std::endl;
                break;
            case '*':
                std::cout << "Result: " << num1 * num2 << std::endl;
                break;
            case '/':
                if (num2 != 0) {
                    std::cout << "Result: " << num1 / num2 << std::endl;
                } else {
                    std::cout << "Error: Division by zero!" << std::endl;
                }
                break;
            default:
                std::cout << "Error: Invalid operator entered." << std::endl;
                break;
        }
    }

    return 0;
}