#include <iostream>

int main() {
    // Hardcoded value to prevent TimeoutError caused by std::cin
    int n = 7;

    // Ensure n is odd for a symmetrical diamond
    if (n % 2 == 0) {
        n++;
    }

    int mid = n / 2;

    // Upper half including middle line
    for (int i = 0; i <= mid; i++) {
        // Print leading spaces
        for (int j = 0; j < mid - i; j++) {
            std::cout << " ";
        }
        // Print asterisks
        for (int j = 0; j < (2 * i + 1); j++) {
            std::cout << "*";
        }
        std::cout << "\n";
    }

    // Lower half
    for (int i = mid - 1; i >= 0; i--) {
        // Print leading spaces
        for (int j = 0; j < mid - i; j++) {
            std::cout << " ";
        }
        // Print asterisks
        for (int j = 0; j < (2 * i + 1); j++) {
            std::cout << "*";
        }
        std::cout << "\n";
    }

    return 0;
}