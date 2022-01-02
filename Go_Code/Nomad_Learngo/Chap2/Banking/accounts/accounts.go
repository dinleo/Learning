package accounts

import (
	"errors"
	"fmt"
)

// Account struct
type Account struct {
	owner   string
	balance int
}

// NewAccount 주로 쓰는 constructor (=pointer cnstr)
func NewAccount(owner string) *Account {
	acc := Account{owner: owner, balance: 1000}
	return &acc
}

// Deposit 멤버변경을 하므로 receiver 는 pointer 로 설정
func (a *Account) Deposit(amount int) {
	a.balance += amount
}

// Balance 멤버조회만 하므로 receiver 는 pointer 로 안해도 상관없다
func (a Account) Balance() int {
	return a.balance
}

func (a Account) Owner() string {
	return a.owner
}

// Withdraw error handling
func (a *Account) Withdraw(amount int) error {
	if a.balance < amount {
		return errors.New("Can't withdraw")
	}
	a.balance -= amount
	return nil
}

func (a *Account) ChangeOwner(newOwner string) {
	a.owner = newOwner
}

// 기본출력문구 커스터마이징
func (a Account) String() string {
	return fmt.Sprint(a.owner, "'s account.\nHas:", a.balance)
}
