package lab

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

// Logger is a wrapper of standard log.Logger
type Logger struct {
	*log.Logger
	logFile *os.File
	slackURL string
}

type logOptions struct{
	logFile string
	slackURL string
}

type LoggerOption func(*logOptions)
func defaultLogOptions() *logOptions{
	return &logOptions{
		logFile: "",
		slackURL: "",
	}
}

func WithLoggerFile(file string) LoggerOption{
	return func(o *logOptions){
		o.logFile = file
	}
}

func WithLoggerSlackURL(url string) LoggerOption{
	return func(o *logOptions){
		o.slackURL = url
	}
}

// NewLogger creates a new logger of combined stdout and file logger.
func NewLogger(opts ...LoggerOption) (*Logger, error) {
	var (
		f   *os.File
		err error
	)

	options := defaultLogOptions()
	for _, o := range opts{
		o(options)
	}

	if options.logFile !=""{
		logFile := options.logFile
		f, err = os.OpenFile(logFile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
		if err != nil {
			err = fmt.Errorf("Openning log file (%q) failed: %w", logFile, err)
			return nil, err
		}
	}

	logger := log.Default()
	logger.SetFlags(0)
	if f == nil {
		logger.SetOutput(os.Stdout)
	} else {
		mw := io.MultiWriter(os.Stdout, f)
		logger.SetOutput(mw)
	}

	l := &Logger{logger, f, options.slackURL}
	return l, nil
}

// Close closes logger file.
func (l *Logger) Close() {
	l.logFile.Close()
}

// HasSlack returns whether logger has Slack webhook url
func(l *Logger) HasSlack() bool{
	return l.slackURL != ""
}

type SlackRequestBody struct {
    Text string `json:"text"`
}

// SendSlack send notification to Slack if being configured.
func(l *Logger) SendSlack(msg string) error{
	if !l.HasSlack(){
		err := fmt.Errorf("No Slack webhook URL configured for the logger.")
		return err
	}

	slackBody, _ := json.Marshal(SlackRequestBody{Text: msg})
	req, err := http.NewRequest(http.MethodPost, l.slackURL, bytes.NewBuffer(slackBody))
	if err != nil {
			return err
	}

	req.Header.Add("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
			return err
	}

	buf := new(bytes.Buffer)
	buf.ReadFrom(resp.Body)
	if buf.String() != "ok" {
		err := fmt.Errorf("logger.SendSlack failed: Non-ok response returned from Slack")
		return err
	}
	return nil
}
