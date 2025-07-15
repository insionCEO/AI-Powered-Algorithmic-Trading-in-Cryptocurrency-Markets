import { createChart } from 'lightweight-charts';
import React, { useEffect, useState, useRef } from 'react';
import './TradingViewChart.css';
import { Typography, Box } from '@mui/material';

const PortfolioGraph = (props) => {

	const chartContainerRef = useRef();

	const [cashSeries, setCashSeries] = useState(null);
	const [commissionSeries, setcommissionSeries] = useState(null);
	const [totalSeries, settotalSeries] = useState(null);

	const resetChartData = () => {
		cashSeries.setData([]);
		commissionSeries.setData([]);
		totalSeries.setData([]);
	};

	useEffect(() => {

		if (props.reset === true)
			resetChartData();

		if (cashSeries && props.cashHoldings.length > 0)
		{

			cashSeries.update(props.cashHoldings.at(-1));

		}
		if (commissionSeries)
		{

			commissionSeries.setData(props.commissionHoldings);

		}
		if (totalSeries)
		{

			totalSeries.setData(props.totalHoldings);

		}
	}, [props.cashHoldings, props.commissionHoldings, props.totalHoldings]);

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
	
		const cashChart = chart.addLineSeries({color: "blue"});
		const commissionChart = chart.addLineSeries({color: "red"});
		const totalChart = chart.addLineSeries({color: "green"});

		cashChart.setData(props.cashHoldings);
		commissionChart.setData(props.commissionHoldings);
		totalChart.setData(props.totalHoldings);
		
		setCashSeries(cashChart);
		setcommissionSeries(commissionChart);
		settotalSeries(totalChart);

		return () => {
			chart.remove();
		};
	}, [])

	return (
		<>
			<div className="chart-container" ref={chartContainerRef} style={{ position: "relative" }}>
				<div style={{ position: "absolute", left: 12, top: 12, zIndex: 25, display: "flex", flexDirection: "column" }}>
					<Box sx={{ display: "flex", alignItems: "center" }}>
						<Typography variant="h4" style={{ color: 'white', fontWeight: 'bold', marginRight: 10 }}>Total Holdings</Typography>
						<Box sx={{ height: "12px", width: "12px", backgroundColor: "green", Border: "1px solid black" }}/>
					</Box>
					<Box sx={{ display: "flex", alignItems: "center" }}>
						<Typography variant="h5" style={{ color: 'rgba(220,220,220,1)', fontWeight: 'normal', marginRight: 10 }}>Cash Holdings</Typography>
						<Box sx={{ height: "12px", width: "12px", backgroundColor: "blue", Border: "1px solid black" }}/>
					</Box>
					<Box sx={{ display: "flex", alignItems: "center" }}>
						<Typography variant="body1" style={{ color: 'rgba(200,200,200,1)', fontWeight: 'normal', marginRight: 10 }}>Commission</Typography>
						<Box sx={{ height: "12px", width: "12px", backgroundColor: "red", Border: "1px solid black" }}/>
					</Box>
				</div>
			</div>
		</>
	)
}

export default PortfolioGraph