import React, { useEffect, useState } from 'react';
import { Grid, Paper, Box } from '@mui/material/';
import Portfolio from './Portfolio';
import Strategy from './Strategy';
import TradingViewChart from './TradingViewChart';
import StrategyHistory from './StrategyHistory';

const Dashboard = (props) => {

	// Dataset
	const [dataset, setDataSet] = useState("sample-1");
	const datasets = [
		{value: "sample-1", label: "Sample 1"},
		{value: "sample-2", label: "Sample 2"},
		{value: "sample-3", label: "Sample 3"}
	];

	// WebSocket
	const [websocket, setWebsocket] = useState(null);

	// Trading Graph
	const [priceData, setPricesData] = useState([]);
	const [predictionData, setPredictionData] = useState([]);
	
	// Portfolio
	const [cashHoldings, setCashHoldings] = useState([]);
	const [btcHoldings, setBtcHoldings] = useState([]);
	const [btcPosition, setBtcPosition] = useState([]);
	const [commissionHoldings, setCommissionHoldings] = useState([]);
	const [totalHoldings, setTotalHoldings] = useState([]);
	
	// Strategy History
	const [fillData, setFillData] = useState([]);
	const [markers, setMarkers] = useState([]);
	const [strategyHistory, setStrategyHistory] = useState([]);

	const [reset, setReset] = useState(false);
	
	const onConnect = () => {

		if (websocket)
		{

			websocket.close();

		}

		const socket = new WebSocket('ws://localhost:8765');

		socket.onopen = () => {
			console.log("Connected to server.");
			const mes = { 
				"type": "connect",
				"dataset": dataset,
			};
			socket.send(JSON.stringify(mes));
		};

		setWebsocket(socket);

		setPricesData([]);
		setPredictionData([]);
	
		setCashHoldings([]);
		setBtcHoldings([]);
		setBtcPosition([]);
		setCommissionHoldings([]);
		setTotalHoldings([]);
		
		setFillData([]);
		setMarkers([]);

		flipReset();


		// TODO disconnect websocket not just create new one
		return () => {
			socket.close();
		};


	};

	useEffect(() => {
		if (websocket)
		{
			websocket.onmessage = (event) => {
				const message = JSON.parse(event.data);
				console.log(message);
				if (message.type === "market_and_portofolio")
				{
					const inPrice = { time: message.timestamp / 1000, value: message.midpoint };
					setPricesData((prevPrices) => {
						const newData = [...prevPrices, inPrice];
						const sortData = newData.sort((a, b) => a.time - b.time);
						return sortData;
					});

					if (message.hasOwnProperty('cash_holdings'))
					{
						const inCashHoldings = { time: message.timestamp / 1000, value: message.cash_holdings };
						setCashHoldings((prevCash) => {
							const newData = [...prevCash, inCashHoldings];
							const sortData = newData.sort((a, b) => a.time - b.time);
							return sortData;
						});
					}

					if (message.hasOwnProperty('btc_holdings'))
					{
						const inBtcHoldings = { time: message.timestamp / 1000, value: message.btc_holdings };
						setBtcHoldings((prevBtcHoldings) => {
							const newData = [...prevBtcHoldings, inBtcHoldings];
							const sortData = newData.sort((a, b) => a.time - b.time);
							return sortData;
						});
					}

					if (message.hasOwnProperty('btc_position'))
					{
						// const inBtcPosition = { time: message.timestamp / 1000, value: message.btc_position };
						// setBtcPosition((prevBtcPositions) => {
						// 	const newData = [...prevBtcPositions, inBtcPosition];
						// 	const sortData = newData.sort((a, b) => a.time - b.time);
						// 	return sortData;
						// });

						const inBtcPosition = { time: message.timestamp / 1000, value: message.btc_position };
						setBtcPosition((prevBtcPositions) => {
							const existingIndex = prevBtcPositions.findIndex(position => position.time === inBtcPosition.time);
							if (existingIndex !== -1) {
								// Replace the existing position with the new position
								prevBtcPositions[existingIndex] = inBtcPosition;
								return [...prevBtcPositions];
							} else {
								// Add the new position to the array and sort by time
								const newData = [...prevBtcPositions, inBtcPosition];
								const sortData = newData.sort((a, b) => a.time - b.time);
								return sortData;
							}
						});

					}

					if (message.hasOwnProperty('commission_holdings'))
					{
						// const incommissionHoldings = { time: message.timestamp / 1000, value: message.commission_holdings };
						// setCommissionHoldings((prevcommission) => {
						// 	const newData = [...prevcommission, incommissionHoldings];
						// 	const sortData = newData.sort((a, b) => a.time - b.time);
						// 	return sortData;
						// });

						const inCommissionHoldings = { time: message.timestamp / 1000, value: message.commission_holdings };
						setCommissionHoldings((prevCommission) => {
							const existingData = prevCommission.find(data => data.time === inCommissionHoldings.time);
							if (existingData) {
								// If data with the same timestamp / 1000 already exists, replace it with the new data
								return prevCommission.map(data => data.time === inCommissionHoldings.time ? inCommissionHoldings : data);
							} else {
								// If data with the same timestamp / 1000 does not exist, add the new data and sort by timestamp / 1000
								const newData = [...prevCommission, inCommissionHoldings];
								const sortedData = newData.sort((a, b) => a.time - b.time);
								return sortedData;
							}
						});


					}

					if (message.hasOwnProperty('total_holdings'))
					{
						// const inTotalHoldings = { time: message.timestamp / 1000, value: message.total_holdings };
						// setTotalHoldings((prevTotal) => {
						// 	const newData = [...prevTotal, inTotalHoldings];
						// 	const sortData = newData.sort((a, b) => a.time - b.time);
						// 	return sortData;
						// });

						const inTotalHoldings = { time: message.timestamp / 1000, value: message.total_holdings };
						setTotalHoldings((prevTotal) => {
							const index = prevTotal.findIndex((data) => data.time === inTotalHoldings.time);
							if (index === -1) { // If no existing data has the same time, add it to the array
								const newData = [...prevTotal, inTotalHoldings];
								const sortData = newData.sort((a, b) => a.time - b.time);
								return sortData;
							} else { // If an existing data has the same time, update the existing data with the new value
								const newData = [...prevTotal];
								newData[index] = inTotalHoldings;
								return newData;
							}
						});

					}
				}
				else if (message.type === "price_prediction")
				{

					const inPricePrediction = { time: message.timestamp_prediction / 1000, value: message.price_prediction };
					setPredictionData((prevPrediction) => {
						return [...prevPrediction, inPricePrediction];
					});

				}
				else if (message.type === "fill")
				{

					const inFill = { time: message.timestamp / 1000, traded_price: message.traded_price, quantity: message.quantity, direction: message.direction, fill_cost: message.fill_cost, commission: message.commission };
					setFillData((prevFill) => {
						return [...prevFill, inFill];
					})
					var inMarker = {};
					if (inFill.direction === 'BUY')
					{

						inMarker = {
								time: inFill.time,
								position: 'belowBar',
								color: 'green',
								shape: 'arrowUp',
								text: 'Buy @ ' + inFill.traded_price,
							};

					}
					else
					{

						inMarker = {
							time: inFill.time,
							position: 'aboveBar',
							color: 'red',
							shape: 'arrowDown',
							text: 'Sell @ ' + inFill.traded_price,
						};

					}
					
					setMarkers((prevMarkers) => {
						return [...prevMarkers, inMarker];
					});

				}

			};
		}
		else
		{

			console.log("Websocket is null");

		}
	}, [websocket])

	const onDatasetChange = (event) => {
		setDataSet(event.target.value);
	};

	const flipReset = () => {
		setReset(!reset);
	};

	const onStop = (event) => {
		if (websocket)
		{

			var message = {
				"type": "stop_trading"
			};

			websocket.send(JSON.stringify(message));
			console.log(message);

			if (fillData.length > 0)
			{

				const newData = { fill: fillData, markers: markers };

				setStrategyHistory(prevData => [...prevData, newData]);
				
				setFillData([]);
				setMarkers([]);

			}

		}
	}

	return (
		<Grid container spacing={2} padding={{ xs: 1, md: 2}} sx={{ height: "100vh", display: "flex"}}>
			<Grid item xs={12} sx={{ minHeight: "75vh"}}>
				<Paper variant='outlined' sx={{ height: "75vh", display: "flex", justifyContent: "center", alignItems: "center", margin: 2 }}>
					<TradingViewChart priceData={priceData} predictionData={predictionData} markers={markers} flipReset={flipReset} reset={reset}/>
				</Paper>
			</Grid>
			<Grid item xs={12} sx={{ display: "flex"}}>
				<Portfolio datasets={datasets} dataset={dataset} onDatasetChange={onDatasetChange} onConnect={onConnect} cashHoldings={cashHoldings} btcHoldings={btcHoldings} btcPosition={btcPosition} commissionHoldings={commissionHoldings} totalHoldings={totalHoldings} reset={reset} />
			</Grid>
			<Grid item xs={12}>
				<Paper variant='outlined'>
					<Box p={{ xs: 1, md: 2 }}>
						<Strategy websocket={websocket} onStop={onStop} />
					</Box>
				</Paper>
			</Grid>
			<Grid item xs={12}>
				<Paper variant='outlined'>
					<Box p={{ xs: 1, md: 2 }}>
						<StrategyHistory strategyHistory={strategyHistory} fillData={fillData} />
					</Box>
				</Paper>
			</Grid>
		</Grid>
  	);
}

export default Dashboard;
