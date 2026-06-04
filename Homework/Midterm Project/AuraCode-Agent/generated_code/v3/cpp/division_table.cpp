#include <iostream>
#include <iomanip>
#include <string>

int main() {
    // ANSI Color codes for different columns to make it visually appealing
    const std::string colors[] = {
        "\033[31m", // Red
        "\033[32m", // Green
        "\033[33m", // Yellow
        "\033[34m", // Blue
        "\033[35m", // Magenta
        "\033[36m", // Cyan
        "\033[37m"  // White
    };
    const std::string reset = "\033[0m";

    std::cout << "Division Table (1 to 20)" << std::endl;
    std::cout << "Format: Dividend / Divisor = Result" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int i = 1; i <= 20; ++i) {
        // Cycle through colors based on the row index
        std::string currentColor = colors[i % 7];
        
        for (int j = 1; j <= 20; ++j) {
            double result = static_cast<double>(i) / j;
            
            std::cout << currentColor;
            std::cout << i << "/" << j << "=" << std::fixed << std::setprecision(2) << result;
            std::cout << reset << "  |  ";
            
            // Wrap line every 5 elements for readability
            if (j % 5 == 0) {
                std::cout << "\n";
            }
        }
        std::cout << "------------------------------------------------------------" << std::endl;
    }

    return 0;
}