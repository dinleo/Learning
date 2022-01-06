package main

import (
	"encoding/csv"
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

var baseURL = "https://kr.indeed.com/jobs?q=python&limit=50"

func main() {
	var jobs []jobInfo
	totalPages := getPages()
	for i := 0; i < totalPages; i++ {
		jobs = append(jobs, getPage(i)...)
	}
	writeJobs(jobs)
}

func writeJobs(jobs []jobInfo) {
	file, err := os.Create("Nomad_Learngo/Chap4/JobScrapper/jobs.csv")
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

func getPage(page int) []jobInfo {
	var jobs []jobInfo
	pageUrl := baseURL + "&start=" + strconv.Itoa(page*50)
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
		job := extractJobInfo(selection)
		jobs = append(jobs, job)
	})

	return jobs
}

func extractJobInfo(selection *goquery.Selection) jobInfo {
	id, _ := selection.Attr("data-jk")
	title := selection.Find(".jobTitle>span").Text()
	location := selection.Find(".companyLocation").Text()
	company := selection.Find(".companyName").Text()
	return jobInfo{
		id:       id,
		title:    title,
		location: location,
		company:  company,
	}
}

func getPages() int {
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
