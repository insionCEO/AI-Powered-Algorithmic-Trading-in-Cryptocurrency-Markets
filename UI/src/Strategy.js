import { InputLabel, Select, FormControl, MenuItem, Box, Typography, Button } from '@mui/material'
import React from 'react'
import { useState } from 'react';
import Selector from './Selector';

const Strategy = (props) => {

	const [model, setModel] = useState("pyraformer");
	const [tradingMethod, setTradingMethod] = useState("heuristics");
	
	const models = [
		{value: "informer", label: "Informer"},
		{value: "transformer", label: "Transformer"},
		{value: "pyraformer", label: "Pyraformer"}
	]

	const tradingMethods = [
		{value: "heuristics", label: "Heuristics"},
		{value: "drl", label: "Deep Reinforcement Learning"}
	]
	
	const onModelChange = (event) => {
		setModel(event.target.value);
	};

	const onTradingMethodChange = (event) => {
		setTradingMethod(event.target.value);
	};

	const onExecute = (event) => {
		if (props.websocket)
		{
			var message = { 
				"type": "trade",
				"alpha": model,
				"trading_decision_making": tradingMethod
			};

			props.websocket.send(JSON.stringify(message));

			console.log(message);
		}
		else
		{

			console.log("Websocket is null");

		}
	};

	

	return (
		<div style={{ display: "flex", justifyContent: "space-around", alignItems: "center" }}>
			<Selector
			options={models}
			label="Select Model"
			value={model}
			onChange={onModelChange}
			/>
			<Selector
			options={tradingMethods}
			label="Select Trading Decision Making"
			value={tradingMethod}
			onChange={onTradingMethodChange}
			width={200}
			/>
			<Button variant="outlined" color="success" onClick={onExecute}>
				Execute
			</Button>
			<Button variant='outlined' color="error" onClick={props.onStop}>
				Stop
			</Button>
		</div>
	);
};

export default Strategy