import { createChart } from 'lightweight-charts';
import React, { useEffect, useState, useRef } from 'react';
import './TradingViewChart.css';
import { Typography, Box } from '@mui/material';

const BitcoinGraph = (props) => {

	const chartContainerRef = useRef();

	const [btcHoldingsSeries, setbtcHoldingsSeries] = useState(null);
	const [btcPositionSeries, setbtcPositionSeries] = useState(null);

	const resetChartData = () => {
		btcHoldingsSeries.setData([]);
		btcPositionSeries.setData([]);
	};

	useEffect(() => {

		if (props.reset === true)
			resetChartData();

		if (btcHoldingsSeries && props.btcHoldings.length > 0)
		{

			btcHoldingsSeries.update(props.btcHoldings.at(-1));

		}
		if (btcPositionSeries)
		{

			btcPositionSeries.setData(props.btcPosition);

		}

	}, [props.btcHoldings, props.btcPosition]);

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
	
		const btcHoldingsChart = chart.addLineSeries({color: "orange"});
		const btcPositionChart = chart.addLineSeries({color: "yellow"});

		btcHoldingsChart.setData(props.btcHoldings);
		btcPositionChart.setData(props.btcPosition);
		
		setbtcHoldingsSeries(btcHoldingsChart);
		setbtcPositionSeries(btcPositionChart);

		return () => chart.remove();
	}, [])

	return (
		<div className="chart-container" ref={chartContainerRef} style={{ position: "relative" }}>
			<div style={{ position: "absolute", left: 12, top: 12, zIndex: 25, display: "flex", flexDirection: "column"}}>
				<Box sx={{ display: "flex", alignItems: "center" }}>
					<Typography variant="h5" style={{ color: 'white', fontWeight: 'bold', marginRight: 10 }}>Bitcoin Holdings</Typography>
					<Box sx={{ height: "12px", width: "12px", backgroundColor: "orange", Border: "1px solid black" }}/>
				</Box>
				<Box sx={{ display: "flex", alignItems: "center" }}>
					<Typography variant="h6" style={{ color: 'rgba(200,200,200,1)', fontWeight: 'normal', marginRight: 10 }}>Bitcoin Position</Typography>
					<Box sx={{ height: "12px", width: "12px", backgroundColor: "yellow", Border: "1px solid black" }}/>
				</Box>
			</div>
		</div>
	)
}

export default BitcoinGraph