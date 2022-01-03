package mydict

import "errors"

type Dictionary map[string]int

var errNotFound = errors.New("Not Found")
var errWordExists = errors.New("Word already exists")

func (d Dictionary) Search(word string) (int, error) {
	value, exists := d[word]
	if exists {
		return value, nil
	}
	return 0, errNotFound
}

func (d Dictionary) Add(word string, num int) error {
	_, err := d.Search(word)
	if err == errNotFound{
		d[word] = num
		return nil
	}
	return errWordExists
}
