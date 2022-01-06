package scrapper

import (
	"encoding/csv"
	"fmt"
	"github.com/PuerkitoBio/goquery"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
)

type jobInfo struct {
	id       string
	title    string
	location string
	company  string
}

// Scarpe indeed by term
func Scrape(term string) {
	var baseURL = "https://kr.indeed.com/jobs?q=" + term + "&limit=50"
	var jobs []jobInfo

	mainC := make(chan []jobInfo)

	totalPages := getPages(baseURL)
	for i := 0; i < totalPages; i++ {
		go getPage(baseURL, i, mainC)
	}
	for i := 0; i < totalPages; i++ {
		job := <-mainC
		jobs = append(jobs, job...)
	}
	writeJobs(jobs)
}

func writeJobs(jobs []jobInfo) {
	file, err := os.Create("Nomad_Learngo/Chap5/scrapper/jobs.csv")
	checkErr(err)

	utf8bom := []byte{0xEF, 0xBB, 0xBF}
	_, _ = file.Write(utf8bom)

	w := csv.NewWriter(file)
	defer w.Flush()

	headers := []string{"Link", "Title", "Location", "Company"}

	wErr := w.Write(headers)
	checkErr(wErr)

	for _, j := range jobs {
		job := []string{"https://kr.indeed.com/viewjob?jk=" + j.id, j.title, j.location, j.company}
		jwErr := w.Write(job)
		checkErr(jwErr)
	}
}

func getPage(baseURL string, page int, mainC chan<- []jobInfo) {
	var jobs []jobInfo

	c := make(chan jobInfo)

	pageUrl := baseURL + "&start=" + strconv.Itoa(page*50)
	fmt.Println("Requesting: ", pageUrl)
	res, err := http.Get(pageUrl)
	checkErr(err)
	checkRes(res)

	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {

		}
	}(res.Body)

	doc, err := goquery.NewDocumentFromReader(res.Body)
	checkErr(err)

	searchCards := doc.Find(".tapItem")

	searchCards.Each(func(i int, selection *goquery.Selection) {
		go extractJobInfo(selection, c)
	})
	for i := 0; i < searchCards.Length(); i++ {
		job := <-c
		jobs = append(jobs, job)
	}

	mainC <- jobs
}

func extractJobInfo(selection *goquery.Selection, c chan<- jobInfo) {
	id, _ := selection.Attr("data-jk")
	title := selection.Find(".jobTitle>span").Text()
	location := selection.Find(".companyLocation").Text()
	company := selection.Find(".companyName").Text()
	c <- jobInfo{
		id:       id,
		title:    title,
		location: location,
		company:  company,
	}
}

func getPages(baseURL string) int {
	pages := 0
	res, err := http.Get(baseURL)
	checkErr(err)
	checkRes(res)

	defer func(Body io.ReadCloser) {
		err := Body.Close()
		if err != nil {

		}
	}(res.Body)

	doc, err := goquery.NewDocumentFromReader(res.Body)
	checkErr(err)

	doc.Find(".pagination").Each(func(i int, selection *goquery.Selection) {
		pages = selection.Find("a").Length()
	})
	return pages
}

func checkErr(err error) {
	if err != nil {
		log.Fatalln(err)
	}
}

func checkRes(res *http.Response) {
	if res.StatusCode != 200 {
		log.Fatalf("status code error: %d %s\n", res.StatusCode, res.Status)
	}
}
