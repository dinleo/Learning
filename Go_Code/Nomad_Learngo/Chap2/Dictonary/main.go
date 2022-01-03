package main

import (
	"Learning/Nomad_Learngo/Chap2/Dictonary/mydict"
	"fmt"
	"os"
)

func main() {
	var m string
	var w string
	var n int
	var err error
	d := mydict.Dictionary{}

	for {
		fmt.Println(d)
		fmt.Println("메뉴를 선택하세요: a: Add, s: Search, u: Update, d: Delete, e: Exit")
		fmt.Scanln(&m)
		switch m {
		case "a":
			fmt.Print("추가 할 Key: ")
			fmt.Scanln(&w)
			fmt.Print("추가 할 Val: ")
			fmt.Scanln(&n)
			err = d.Add(w, n)
			if err != nil {
				fmt.Println(err)
				break
			} else {
				fmt.Println("Added successfully")
			}
		case "s":
			fmt.Print("검색 할 Key: ")
			fmt.Scanln(&w)
			num, err := d.Search(w)
			if err != nil {
				fmt.Println(err)
				break
			} else {
				fmt.Println(num)
			}
		case "u":
			fmt.Print("수정 할 Key: ")
			fmt.Scanln(&w)
			fmt.Print("수정 할 Val: ")
			fmt.Scanln(&n)
			err = d.Update(w, n)
			if err != nil {
				fmt.Println(err)
				continue
			} else {
				fmt.Println("Updated successfully")
			}
		case "d":
			fmt.Print("삭제 할 Key: ")
			fmt.Scanln(&w)
			err = d.Delete(w)
			if err != nil {
				fmt.Println(err)
				break
			} else {
				fmt.Println("Deleted successfully")
			}
		case "e":
			os.Exit(0)
		default:
			fmt.Println("잘못 입력하셨습니다.")
		}
	}
}
