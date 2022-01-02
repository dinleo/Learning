package main

import "fmt"

func main() {
	a := 2
	b := &a
	*b = 3

	fmt.Println(&a, a, *b)
}
