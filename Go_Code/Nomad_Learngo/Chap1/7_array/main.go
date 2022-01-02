package main

import "fmt"

func main() {
	names := []string{"a", "b", "c"}
	names = append(names, "d")
	fmt.Println(names)
}
