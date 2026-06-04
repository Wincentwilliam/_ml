const generateFibonacci = (terms) => {
    const sequence = [0, 1];
    for (let i = 2; i < terms; i++) {
        sequence.push(sequence[i - 1] + sequence[i - 2]);
    }
    return sequence;
};

const printFibonacciTable = (limit) => {
    const data = generateFibonacci(limit);
    
    // Define table headers
    const headerIndex = "Index";
    const headerValue = "Fibonacci Value";
    
    // Calculate column widths based on the largest number
    const maxValueStr = data[limit - 1].toString();
    const valueWidth = Math.max(headerValue.length, maxValueStr.length);
    const indexWidth = Math.max(headerIndex.length, limit.toString().length);

    const separator = "-".repeat(indexWidth + valueWidth + 3);

    console.log("\n--- Fibonacci Sequence Generation ---");
    console.log(separator);
    console.log(`${headerIndex.padEnd(indexWidth)} | ${headerValue.padEnd(valueWidth)}`);
    console.log(separator);

    data.forEach((val, idx) => {
        const indexStr = idx.toString().padEnd(indexWidth);
        const valueStr = val.toString().padEnd(valueWidth);
        console.log(`${indexStr} | ${valueStr}`);
    });

    console.log(separator);
    console.log(`Total Terms Generated: ${limit}`);
    console.log("Calculation Method: Iterative Addition (F(n) = F(n-1) + F(n-2))\n");
};

printFibonacciTable(30);