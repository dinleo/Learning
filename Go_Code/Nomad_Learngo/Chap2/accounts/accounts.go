package accounts

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
