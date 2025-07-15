import { createChart } from 'lightweight-charts';
import React, { useEffect, useState, useRef } from 'react';
import './TradingViewChart.css';
import { Typography, Box } from '@mui/material';

const TradingViewChart = (props) => {

	const chartContainerRef = useRef();

	const [lineSeries, setLineSeries] = useState(null);
	const [predictionSeries, setPredictionSeries] = useState(null);

	const resetChartData = () => {
		lineSeries.setData([]);
		predictionSeries.setData([]);
	};

	useEffect(() => {

		if (props.reset === true)
		{

			resetChartData();
			props.flipReset();

		}

		if (lineSeries && props.priceData.length > 0)
		{

			lineSeries.update(props.priceData.at(-1));
			lineSeries.setMarkers(props.markers);

		}
		if (predictionSeries && props.predictionData.length > 0)
		{

			predictionSeries.update(props.predictionData.at(-1));

		}

	}, [props.priceData, props.predictionData, props.reset]);
	
	useEffect(() => {

		const chart = createChart(chartContainerRef.current, {
			layout: {
				background: { color: "transparent"},
				textColor: "grey",
			},
			grid: {
				vertLines: {
					color: "grey",
				},
				horzLines: {
					color: "grey",
				},
			},
			width: chartContainerRef.current.clientWidth,
			timeScale: {
				tickMarkColor: "blue",
				lineColor: "red",
				timeVisible: true,
				secondsVisible: true,
				timeUnit: "second",
				borderColor: "transparent"
			},
			rightPriceScale: {
				visible: true,
				borderColor: "transparent"
			},
			autoSize: true,
		});
	
		const lineChart = chart.addLineSeries({ color: "blue" });
		const predictionChart = chart.addLineSeries({ color: "green" });
		if (props.priceData.length > 0)
			lineChart.setData(props.priceData);
		if (props.predictionData > 0)
			predictionChart.setData(props.predictionData);
		setLineSeries(lineChart);
		setPredictionSeries(predictionChart);

		return () => chart.remove();
	}, [])

	return (
		<div className="chart-container" ref={chartContainerRef} style={{ position: "relative" }}>
			<div style={{ position: "absolute", left: 12, top: 12, zIndex: 25, display: "flex", flexDirection: "column"}}>
				<Box sx={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
					<Typography variant="h4" style={{ color: 'white', fontWeight: 'bold', marginRight: 10 }}>Bitcoin/USD</Typography>
					<Box sx={{ height: "16px", width: "16px", backgroundColor: "blue", Border: "1px solid black" }}/>
				</Box>
				<Box sx={{ display: "flex", alignItems: "center" }}>
					<Typography variant="h5" style={{ color: 'rgba(220,220,220,1)', fontWeight: 'normal', marginRight: 10 }}>Predicted Price</Typography>
					<Box sx={{ height: "16px", width: "16px", backgroundColor: "green", Border: "1px solid black" }}/>
				</Box>
			</div>
		</div>
	)
}

export default TradingViewChart