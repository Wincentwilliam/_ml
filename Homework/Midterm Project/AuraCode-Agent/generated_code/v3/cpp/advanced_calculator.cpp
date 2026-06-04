#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>

using namespace std;

// Function to perform calculations based on a choice and inputs
void performCalculation(int choice, double num1, double num2 = 0) {
    cout << fixed << setprecision(4);
    cout << "Choice " << choice << " | Input: " << num1;
    if (choice != 6 && choice != 7 && choice != 8 && choice != 9 && choice != 10 && choice != 11) {
        cout << ", " << num2;
    }
    cout << " -> ";

    switch (choice) {
        case 1:
            cout << "Result: " << (num1 + num2) << endl;
            break;
        case 2:
            cout << "Result: " << (num1 - num2) << endl;
            break;
        case 3:
            cout << "Result: " << (num1 * num2) << endl;
            break;
        case 4:
            if (num2 == 0) cout << "Error: Division by zero!" << endl;
            else cout << "Result: " << (num1 / num2) << endl;
            break;
        case 5:
            cout << "Result: " << pow(num1, num2) << endl;
            break;
        case 6:
            if (num1 < 0) cout << "Error: Negative input for square root!" << endl;
            else cout << "Result: " << sqrt(num1) << endl;
            break;
        case 7:
            cout << "Result: " << sin(num1 * M_PI / 180.0) << endl;
            break;
        case 8:
            cout << "Result: " << cos(num1 * M_PI / 180.0) << endl;
            break;
        case 9:
            cout << "Result: " << tan(num1 * M_PI / 180.0) << endl;
            break;
        case 10:
            if (num1 <= 0) cout << "Error: Logarithm of non-positive number!" << endl;
            else cout << "Result: " << log(num1) << endl;
            break;
        case 11:
            if (num1 <= 0) cout << "Error: Logarithm of non-positive number!" << endl;
            else cout << "Result: " << log10(num1) << endl;
            break;
        default:
            cout << "Invalid choice!" << endl;
    }
}

int main() {
    cout << "====================================" << endl;
    cout << "   ADVANCED C++ CALCULATOR DEMO      " << endl;
    cout << "====================================" << endl;

    // Hardcoded test cases to avoid TimeoutError from cin
    performCalculation(1, 10.5, 5.2);   // Addition
    performCalculation(2, 10.5, 5.2);   // Subtraction
    performCalculation(3, 10.5, 5.2);   // Multiplication
    performCalculation(4, 10.5, 5.2);   // Division
    performCalculation(5, 2.0, 3.0);    // Power (2^3)
    performCalculation(6, 16.0);         // Square Root
    performCalculation(7, 90.0);        // Sine 90 deg
    performCalculation(8, 0.0);          // Cosine 0 deg
    performCalculation(9, 45.0);        // Tangent 45 deg
    performCalculation(10, 2.71828);    // Natural Log (approx e)
    performCalculation(11, 100.0);      // Common Log (log10 100)
    
    // Error case tests
    performCalculation(4, 10.0, 0.0);    // Div by zero
    performCalculation(6, -1.0);         // Sqrt negative
    performCalculation(10, -5.0);        // Log negative

    cout << "====================================" << endl;
    cout << "Demo completed successfully." << endl;

    return 0;
}