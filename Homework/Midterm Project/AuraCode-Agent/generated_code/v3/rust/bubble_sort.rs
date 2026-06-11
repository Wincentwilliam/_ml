fn bubble_sort(arr: &mut [i32]) {
    let n = arr.len();
    for i in 0..n {
        let mut swapped = false;
        for j in 0..n - 1 - i {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
                swapped = true;
            }
        }
        // If no two elements were swapped by inner loop, then break
        if !swapped {
            break;
        }
    }
}

fn main() {
    let mut numbers = vec![64, 34, 25, 12, 22, 11, 90];
    
    println!("Original array: {:?}", numbers);
    
    bubble_sort(&mut numbers);
    
    println!("Sorted array:   {:?}", numbers);
}