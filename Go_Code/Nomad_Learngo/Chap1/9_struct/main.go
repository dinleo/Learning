package main

import "fmt"

type person struct {
	name    string
	age     int
	FavFood []string
}

func main() {
	fav := []string{"김치", "라면"}
	leo := person{
		name:    "leo",
		age:     20,
		FavFood: fav,
	}

	fmt.Println(leo.FavFood)
}
