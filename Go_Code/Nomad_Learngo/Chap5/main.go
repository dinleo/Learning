package main

import (
	"Learning/Nomad_Learngo/Chap5/scrapper"
	"github.com/labstack/echo/v4"
	"os"
	"strings"
)

func handleHOme(c echo.Context) error {
	return c.File("C:\\Users\\dinle\\Code\\Learning\\Go_Code\\Nomad_Learngo\\Chap5\\home.html")
}

func handleScrape(c echo.Context) error {
	defer os.Remove("jobs.csv")
	term := strings.ToLower(scrapper.CleanString(c.FormValue("term")))
	scrapper.Scrape(term)
	return c.Attachment("C:\\Users\\dinle\\Code\\Learning\\Go_Code\\Nomad_Learngo\\Chap5\\jobs.csv", "job.csv")
}

func main() {
	e := echo.New()
	e.GET("/", handleHOme)
	e.POST("/scrape", handleScrape)
	e.Logger.Fatal(e.Start(":1323"))
}
