package main

import (
	"fmt"
	"time"
)

func main() {
	c := make(chan string)
	people := [2]string{"leo", "john"}

	for _, person := range people {
		go isSexy(person, c)
	}
	result := <-c
	fmt.Println(result)
	fmt.Println(<-c)

}

func isSexy(person string, c chan string) {
	fmt.Println(person, "is sexy")
	time.Sleep(time.Second)
	c <- person + "is sexy"
}
