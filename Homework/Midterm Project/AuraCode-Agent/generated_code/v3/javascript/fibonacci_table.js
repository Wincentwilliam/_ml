const generateFibonacci = (terms) => {
    const sequence = [0, 1];
    for (let i = 2; i < terms; i++) {
        sequence.push(sequence[i - 1] + sequence[i - 2]);
    }
    return sequence;
};

const formatAsTable = (sequence) => {
    const tableData = sequence.map((value, index) => ({
        "Term": index + 1,
        "Value": value
    }));
    
    console.log("Fibonacci Sequence (First 20 Terms):");
    console.table(tableData);
};

const termsCount = 20;
const fibSequence = generateFibonacci(termsCount);
formatAsTable(fibSequence);