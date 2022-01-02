package main

import "fmt"

func canIDrink(age int) bool {
	if KoreanAge := age + 2; KoreanAge < 19 {
		return false
	} else {
		return true
	}
}

func canIDreink_switch(age int) bool {
	switch {
	case age < 17:
		return false
	case age == 18:
		return true
	default:
		return true
	}
}

func main() {
	fmt.Println(canIDrink(16))
	fmt.Println(canIDreink_switch(19))
}
