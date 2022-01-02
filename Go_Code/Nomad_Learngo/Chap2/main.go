package main

import (
	"Nomad_Learngo/Chap2/accounts"
	"fmt"
)

func main() {
	account1 := accounts.NewAccount("leo")
	account2 := accounts.Account{}
	fmt.Println(account1, account2)
	account1.Deposit(100)
	account2.Deposit(100)
	fmt.Println(account1, account2)
}
