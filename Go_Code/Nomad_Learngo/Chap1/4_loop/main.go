package main

import "fmt"

func superAdd(num ...int) int {
	total := 0
	for i, n := range num {
		fmt.Println(i, n)
		total += n
	}
	return total
}

func main() {
	total := superAdd(2, 3, 4, 5, 6)
	fmt.Println(total)
}
