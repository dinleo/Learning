package mydict

import "errors"

type Dictionary map[string]int

var errNotFound = errors.New("Key Not Found")
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
	if err == errNotFound {
		d[word] = num
		return nil
	}
	return errWordExists
}

func (d Dictionary) Update(word string, num int) error {
	_, err := d.Search(word)
	if err == nil {
		d[word] = num
		return nil
	}
	return err
}

func (d Dictionary) Delete(word string) error {
	_, exists := d[word]
	if exists {
		delete(d, word)
		return nil
	}
	return errNotFound
}
