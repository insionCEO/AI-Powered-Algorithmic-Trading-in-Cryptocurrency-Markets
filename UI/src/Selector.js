import { InputLabel, Select, FormControl, MenuItem, OutlinedInput } from '@mui/material'
import React from 'react'

const Selector = (props) => {

	return (
		<FormControl sx={{ minWidth: props.width }}>
			<InputLabel id={props.value}>{props.label}</InputLabel>
			<Select
			labelId={props.label}
			value={props.value}
			onChange={props.onChange}
			label={props.label}
			>
				{props.options.map((option) => (
				<MenuItem key={option.value} value={option.value}>
					{option.label}
				</MenuItem>
				))}
			</Select>
		</FormControl>
	);
};

export default Selector