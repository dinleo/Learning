package main

import "fmt"

func main() {
	std := map[string]int{"a": 1, "b": 2}
	fmt.Println(std)
	fmt.Println(std["a"], "=========")

	for k, v := range std {
		fmt.Println(k, v)
	}
}
