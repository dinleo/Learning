package main

import (
	"Nomad_Learngo/Nomad_Learngo/Chap2/Dictonary/mydict"
	"fmt"
)

func main() {
	d := mydict.Dictionary{}
	d["a"] = 1
	d["b"] = 2
	num, err := d.Search("c")
	if err != nil {
		fmt.Println(err)
	} else {
		fmt.Println(num)
	}

}
