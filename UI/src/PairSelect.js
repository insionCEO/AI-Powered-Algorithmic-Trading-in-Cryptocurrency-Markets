import React from 'react';
import { Select, MenuItem, FormControl, InputLabel } from '@mui/material';

const PairSelect = ({ options, label, value, onChange }) => {

  	return (
		<FormControl>
			<InputLabel id="crypto-select-label">{label}</InputLabel>
			<Select
				labelId="crypto-select-label"
				value={value}
				onChange={onChange}
			>
				{options.map((option) => (
				<MenuItem key={option.value} value={option.value}>
					{option.label}
				</MenuItem>
				))}
			</Select>
		</FormControl>
  	);
};

export default PairSelect;
