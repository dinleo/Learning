package main

import (
	"fmt"
	"time"
)

func main() {
	go sexyCount("leo")
	go sexyCount("neo")
	time.Sleep(time.Second * 5)
}
func sexyCount(person string) {
	for i := 0; i < 10; i++ {
		fmt.Println(person, "is sexy", i)
		time.Sleep(time.Second)
	}
}
