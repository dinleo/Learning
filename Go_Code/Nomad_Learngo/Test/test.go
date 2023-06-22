package Nomad_Learngo

import (
	"crypto/sha256"
	"fmt"
)

func main() {
	fmt.Println("Hello 의 해쉬화:" + Hash("Hello"))
}

func Hash(i interface{}) string {
	s := fmt.Sprintf("%v", i)
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(s)))
	return hash
}
