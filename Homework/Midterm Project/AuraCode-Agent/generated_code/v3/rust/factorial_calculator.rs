fn factorial(n: u64) -> u64 {
    let mut result = 1;
    for i in 1..=n {
        result *= i;
    }
    result
}

fn main() {
    println!("Factorials from 1 to 10:");
    println!("-------------------------");
    for i in 1..=10 {
        let res = factorial(i);
        println!("{}! = {}", i, res);
    }
}