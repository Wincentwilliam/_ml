const generateMultiplicationTable = (size) => {
    console.log(`Multiplication Table (1 to ${size}):\n`);
    
    // Create the header row
    let header = "    ";
    for (let i = 1; i <= size; i++) {
        header += i.toString().padStart(4, " ");
    }
    console.log(header);
    console.log("-".repeat(header.length));

    // Generate each row
    for (let i = 1; i <= size; i++) {
        let row = i.toString().padStart(2, " ") + " |";
        for (let j = 1; j <= size; j++) {
            const product = i * j;
            row += product.toString().padStart(4, " ");
        }
        console.log(row);
    }
};

generateMultiplicationTable(10);