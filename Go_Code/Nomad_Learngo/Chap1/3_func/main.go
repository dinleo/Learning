package main

import (
	"fmt"
	"strings"
)

func lenAndUpper(name string) (int, string) {
	defer fmt.Println("lenAndUpper is done")
	return len(name), strings.ToUpper(name)
}

// 복수의 매개변수 받기
func repeat(words ...string) {
	fmt.Println(words)
}

// Naked return
func nake(words string) (leng int, upp string) {
	leng = len(words)
	upp = strings.ToUpper(words)
	return
}

func main() {
	totallen, upperName := lenAndUpper("leo")
	fmt.Println(totallen, upperName)
	repeat("안녕", "친구야", "반가워")
	fmt.Println(nake("허준우"))
}
