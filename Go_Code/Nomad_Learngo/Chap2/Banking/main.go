package main

import (
	"Nomad_Learngo/Nomad_Learngo/Chap2/Banking/accounts"
	"fmt"
)

func main() {
	// 생성
	account1 := accounts.NewAccount("leo")
	account2 := accounts.Account{}
	fmt.Println(account1, account2)

	// 입금
	account1.Deposit(100)
	account2.Deposit(100)
	fmt.Println(account1, account2)

	// 출금
	account1.Withdraw(50)
	fmt.Println(account1)

	// 초과출금 (에러)
	errNoMoney := account1.Withdraw(10000)
	if errNoMoney != nil {
		fmt.Println(errNoMoney)
		// log.Fatalln(errNoMoney)
		// 프로그램 종료시키고 싶을때
	}
	fmt.Println(account1)

	// 오너변경

}
