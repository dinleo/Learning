package main

import (
	"fmt"
	"github.com/PuerkitoBio/goquery"
	"io"
	"log"
	"net/http"
)

const (
	opGG      = "https://www.op.gg/"
	myArticle = "https://www.campuspick.com/mypage/articles?id=myarticle"
	cmpURL    = "https://www.campuspick.com/"
)

func handleRes(res *http.Response) {
	if res.StatusCode != 200 {
		log.Fatalf("status code error: %d %s\n", res.StatusCode, res.Status)
	}
}

func handleErr(err error) {
	if err != nil {
		log.Fatalln(err)
	}
}

func getPages() []string {
	var Pages []string
	res, err := http.Get(opGG)
	handleErr(err)
	handleRes(res)

	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {

		}
	}(res.Body)

	doc, err := goquery.NewDocumentFromReader(res.Body)
	handleErr(err)

	content := doc.Find("div.l-container div.community-best div.community-best__content-left ul")
	content.Each(func(i int, selection *goquery.Selection) {
		page, _ := selection.Attr("class")
		Pages = append(Pages, page)
	})
	return Pages
}

func main() {
	fmt.Println(getPages())
}
