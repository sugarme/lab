package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func makeLossGraphFromCSV(dataDir string) error{
	tloss, err := readLossCSV(dataDir, false)
	if err != nil{
		return err
	}
	vloss, err := readLossCSV(dataDir, true)
	if err != nil{
		return err
	}

	p := plot.New()

	p.Title.Text = "Train/Valid Losses"
	p.X.Label.Text = "Epoch"
	p.Y.Label.Text = "Loss"
	p.Legend.Top = true
	p.Legend.Padding = 5
	p.X.Tick.Marker = EpochTicks{}

	// train points
	trainPoints :=  make(plotter.XYs, len(tloss))
	validPoints := make(plotter.XYs, len(vloss))
	for i := 0; i < len(tloss); i++{
		trainPoints[i].X = float64(i)
		trainPoints[i].Y = tloss[i]
	}

	for i := 0; i < len(vloss); i++{
		validPoints[i].X = float64(i)
		validPoints[i].Y = vloss[i]
	}

	err = plotutil.AddLinePoints(p,
		"Train", trainPoints,
		"Valid", validPoints,
	)
	if err != nil {
		err := fmt.Errorf("Making train loss plot failed: %w\n", err)
		return err
	}

	// Save the plot to a PNG file.
	lossFile := fmt.Sprintf("%s/epoch-loss.png", dataDir)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, lossFile); err != nil {
		err := fmt.Errorf("Saving train valid loss plot failed: %w\n", err)
		return err
	}	

	return nil
}

func readLossCSV(dataDir string, isValid bool)([]float64, error){
	file := fmt.Sprintf("%s/losses.csv", dataDir)
	if isValid{
		file = fmt.Sprintf("%s/valid-losses.csv", dataDir)
	}

	f, err := os.Open(file)
	if err != nil{
		return nil, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)

	isHeader := true
	currEpoch := 0
	var epochLoss []float64
	var losses []float64
	for sc.Scan(){
		// Skip
		if isHeader{
			isHeader = false
			continue
		}	

		line := sc.Text()
		// epoch, step, loss
		fields := strings.Split(line, ",")
		epoch, err :=  strconv.Atoi(fields[0])
		if err != nil{
			err := fmt.Errorf("readLossCSV - convert epoch failed: %w\n", err)
			return nil, err
		}
		loss, err := strconv.ParseFloat(fields[2], 64)
		if err != nil{
			err := fmt.Errorf("readLossCSV - parse loss failed: %w\n", err)
			return nil, err
		}

		// New epoch
		if epoch != currEpoch{
			eloss := stat.Mean(epochLoss, nil)
			losses = append(losses, eloss)
			epochLoss = []float64{}
			epochLoss = append(epochLoss, loss)
			currEpoch = epoch
		} else {
			epochLoss = append(epochLoss, loss)
		}
	}

	return losses, nil
}

type EpochTicks struct{}

// Ticks returns Ticks in the specified range.
func (EpochTicks) Ticks(min, max float64) []plot.Tick {
	if max <= min {
		panic("illegal range")
	}
	var ticks []plot.Tick

	// label every 10 unit
	for i := min; i <= max; i++ {
		if int(i)%5 == 0{
			ticks = append(ticks, plot.Tick{Value: i, Label: strconv.FormatFloat(i, 'f', 0, 64)})
		} else {
			ticks = append(ticks, plot.Tick{Value: i, Label: ""})
		}
	}
	return ticks
}

func makeFindLRGraphFromCSV(dataDir string) error{
	losses, lrs, err := readLRLossCSV(dataDir)
	if err != nil{
		return err
	}

	p := plot.New()

	p.Title.Text = "Learning Rate Finding"
	p.X.Label.Text = "learning rate"
	p.Y.Label.Text = "loss"
	p.Legend.Top = true
	p.Legend.Padding = 5
	p.X.Tick.Marker = LRTicks{}
	p.X.Tick.Label.Rotation = 45
	p.X.Tick.Label.YAlign = draw.YCenter
	p.X.Tick.Label.XAlign = draw.XRight

	// train points
	points :=  make(plotter.XYs, len(losses))
	for i := 0; i < len(losses); i++{
		points[i].X = lrs[i]
		points[i].Y = losses[i]
	}

	err = plotutil.AddLinePoints(p,
		"loss vs lr", points,
	)
	if err != nil {
		err := fmt.Errorf("Making train loss plot failed: %w\n", err)
		return err
	}

	// Save the plot to a PNG file.
	lossFile := fmt.Sprintf("%s/find-lr-from-csv.png", dataDir)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, lossFile); err != nil {
		err := fmt.Errorf("Saving train valid loss plot failed: %w\n", err)
		return err
	}	

	return nil
}

func readLRLossCSV(dataDir string)([]float64, []float64, error){
	file := fmt.Sprintf("%s/find-lr.csv", dataDir)
	f, err := os.Open(file)
	if err != nil{
		return nil, nil,err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)

	isHeader := true
	var(
		losses []float64
		lrs []float64
	)
	for sc.Scan(){
		// Skip
		if isHeader{
			isHeader = false
			continue
		}	

		line := sc.Text()
		//step,loss,lr
		fields := strings.Split(line, ",")
		loss, err := strconv.ParseFloat(fields[1], 64)
		if err != nil{
			err := fmt.Errorf("readLREpochCSV - parse loss failed: %w\n", err)
			return nil, nil, err
		}
		losses = append(losses, loss)

		lr, err := strconv.ParseFloat(fields[2], 64)
		if err != nil{
			err := fmt.Errorf("readLREpochCSV - convert epoch failed: %w\n", err)
			return nil, nil, err
		}
		lrs = append(lrs, lr)
	}

	return losses, lrs, nil
}

type LRTicks struct{}

// Ticks returns Ticks in the specified range.
func (LRTicks) Ticks(min, max float64) []plot.Tick {
	if max <= min {
		panic("illegal range")
	}
	var ticks []plot.Tick

	// label every 10 unit
	count := 0
	for i := min; i <= max; i += min  {
		switch {
			case count %100 == 0:
				ticks = append(ticks, plot.Tick{Value: i, Label: strconv.FormatFloat(i, 'e', 0, 64)})
			case count %10 == 0:
				ticks = append(ticks, plot.Tick{Value: i, Label: ""})
		}
		count++
	}
	return ticks
}

