#include <iostream>
#include <vector>

void bubbleSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap elements
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void printArray(const std::vector<int>& arr) {
    for (int i = 0; i < arr.size(); i++) {
        std::cout << arr[i] << (i == arr.size() - 1 ? "" : " ");
    }
    std::cout << std::endl;
}

int main() {
    std::vector<int> data = {64, 34, 25, 12, 22, 11, 90, 5, 45, 80};

    std::cout << "Array before sorting: ";
    printArray(data);

    bubbleSort(data);

    std::cout << "Array after sorting:  ";
    printArray(data);

    return 0;
}