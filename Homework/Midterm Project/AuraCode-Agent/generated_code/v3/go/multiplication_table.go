package main

import "fmt"

func main() {
	fmt.Println("Multiplication Table (1 to 5):")
	fmt.Println("-----------------------------")
	for i := 1; i <= 5; i++ {
		for j := 1; j <= 5; j++ {
			fmt.Printf("%d\t", i*j)
		}
		fmt.Println()
	}
}