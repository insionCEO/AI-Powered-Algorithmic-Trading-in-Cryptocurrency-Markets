import { Typography, Box, TableContainer, Table, TableHead, TableRow, TableCell, TableBody } from '@mui/material';
import React, { useState } from 'react'

const StrategyHistory = (props) => {

	const [selectedIndex, setSelectedIndex] = useState(null);

	const handleStrategyClick = (index) => {
		setSelectedIndex(index);
	};

	return (
		<>
		<Box sx={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center" }}>
			<Box sx={{ flexGrow: 1, width: "100%" }}>
				<Typography variant="h3" sx={{ borderBottom: "1px solid rgba(47,47,47,1)", margin: 2 }}>
					Strategy History
				</Typography>
			</Box>
			<Box sx={{ display: "flex", flexGrow: 1, width: "100%", margin: 2,  }}>
				<Box sx={{ flexGrow: 1, borderRight: "1px solid rgba(47,47,47,1)", margin: 2 }}>
					{ props.strategyHistory.map((strat, i) => (
						<Typography 
						key={strat.id}
						variant="h6" 
						sx={{ 
							color: selectedIndex === i ? 'primary.main' : 'rgba(200,200,200,1)',
							textDecoration: selectedIndex === i ? "underline" : "none", 
							fontWeight: selectedIndex === i ? "bold" : "normal",
							cursor: "pointer",
							transition: "all 0.3s",
							"&:hover": {
								color: "primary.light",
								textDecoration: "underline",
								transform: "scale(1.1)"
							},
							textAlign: "center"
						}}
						onClick={() => handleStrategyClick(i)}
						>
							Strategy Run {i + 1}
						</Typography>
					))}
				</Box>
				<Box sx={{ flexGrow: 2, margin: 2}}>
					{ selectedIndex != null &&  ( 
						<TableContainer>
							<Table>
								<TableHead>
									<TableRow>
										<TableCell align='center'>Time</TableCell>
										<TableCell align='center'>Traded Price</TableCell>
										<TableCell align='center'>Quantity</TableCell>
										<TableCell align='center'>Direction</TableCell>
										<TableCell align='center'>Fill Cost</TableCell>
										<TableCell align='center'>Commission</TableCell>
									</TableRow>
								</TableHead>
								<TableBody>
									{ props.strategyHistory[selectedIndex].fill.map((fill, i) => (
										<TableRow key={fill.id}>
											<TableCell align='center'>{new Date(fill.time * 1000).toLocaleString()}</TableCell>
											<TableCell align='center'>{fill.traded_price.toFixed(2)}</TableCell>
											<TableCell align='center'>{fill.quantity.toFixed(2)}</TableCell>
											<TableCell align='center'>{fill.direction}</TableCell>
											<TableCell align='center'>{fill.fill_cost.toFixed(2)}</TableCell>
											<TableCell align='center'>{fill.commission.toFixed(2)}</TableCell>
										</TableRow>
									))}
								</TableBody>
							</Table>
						</TableContainer>
					)}
				</Box>
			</Box>
		</Box>
		</>
	)
}

export default StrategyHistory