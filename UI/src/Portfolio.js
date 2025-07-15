import React from 'react'
import PortfolioGraph from './PortfolioGraph'
import BitcoinGraph from './BitcoinGraph'
import { Typography, Grid, Paper } from '@mui/material'
import DatasetConnect from './DatasetConnect'
import PortfolioMetrics from './PortfolioMetrics'

const Portfolio = (props) => {

	return (
		<>
			<Grid item sx={{ display: "flex", flexDirection: "column" }} xs={4}>
				{/* Left items (datasetconnect, protfolio metrics) */}
				<Grid item sx={{ margin: 2 }}>
					<Paper variant='outlined' sx={{ padding: 2}}>
						<DatasetConnect datasets={props.datasets} dataset={props.dataset} onDatasetChange={props.onDatasetChange} onConnect={props.onConnect}/>
					</Paper>
				</Grid>
				<Grid item sx={{ margin: 2}}>
					<Paper variant='outlined'>
						<PortfolioMetrics totalHoldings={props.totalHoldings} cashHoldings={props.cashHoldings} commissionHoldings={props.commissionHoldings} btcHoldings={props.btcHoldings} btcPosition={props.btcPosition} />
					</Paper>
				</Grid>
			</Grid>
			<Grid item sx={{ display: "flex", flexDirection: "column" }} xs={8}>
				{/* Right items portfolio graphs */}
				<Grid item sx={{ margin: 2, height: "50%" }}>
					<Paper variant='outlined' sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100%" }}>
						<PortfolioGraph sx={{ flexGrow: 1}}cashHoldings={props.cashHoldings} commissionHoldings={props.commissionHoldings} totalHoldings={props.totalHoldings} reset={props.reset} />
					</Paper>
				</Grid>
				<Grid item sx={{ margin: 2, height: "50%" }}>
					<Paper variant='outlined' sx={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100%"}}>
						<BitcoinGraph btcHoldings={props.btcHoldings} btcPosition={props.btcPosition} reset={props.reset} />
					</Paper>
				</Grid>
			</Grid>
		</>
	)
}

export default Portfolio