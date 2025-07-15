import React from 'react'
import { Typography, Paper, Box } from '@mui/material'

const PortfolioMetrics = (props) => {

	return (
		<>
            <Box sx={{ margin: 2 }} aria-label="Portfolio Metrics">
                <Typography variant="h2" sx={{ margin: 4 }}>Portfolio</Typography>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: 4, borderBottom: "1px solid rgba(47,47,47,1)" }}>
                    <Typography style={{ textAlign: "left", whiteSpace: "nowrap" }} aria-label="Total Holdings">
                        Total Holdings
                    </Typography>
                    <Typography aria-label={(props.totalHoldings.at(-1)) ? props.totalHoldings.at(-1).value.toFixed(2) : "0.00"}>
                        {(props.totalHoldings.at(-1)) ? props.totalHoldings.at(-1).value.toFixed(2) : "0.00"}
                    </Typography>
                </Box>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: 4, borderBottom: "1px solid rgba(47,47,47,1)" }}>
                    <Typography style={{ textAlign: "left", whiteSpace: "nowrap" }} aria-label="Cash Holdings">
                        Cash Holdings
                    </Typography>
                    <Typography aria-label={(props.cashHoldings.at(-1)) ? props.cashHoldings.at(-1).value.toFixed(2) : "0.00"}>
                        {(props.cashHoldings.at(-1)) ? props.cashHoldings.at(-1).value.toFixed(2) : "0.00"}
                    </Typography>
                </Box>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: 4, borderBottom: "1px solid rgba(47,47,47,1)" }}>
                    <Typography style={{ textAlign: "left", whiteSpace: "nowrap" }} aria-label="Commission">
                        Commission
                    </Typography>
                    <Typography aria-label={(props.commissionHoldings.at(-1)) ? props.commissionHoldings.at(-1).value.toFixed(2) : "0.00"}>
                        {(props.commissionHoldings.at(-1)) ? props.commissionHoldings.at(-1).value.toFixed(2) : "0.00"}
                    </Typography>
                </Box>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: 4, borderBottom: "1px solid rgba(47,47,47,1)" }}>
                    <Typography style={{ textAlign: "left", whiteSpace: "nowrap" }} aria-label="Bitcoin Holdings">
                        Bitcoin Holdings
                    </Typography>
                    <Typography aria-label={(props.btcHoldings.at(-1)) ? props.btcHoldings.at(-1).value.toFixed(2) : "0.00"}>
                        {(props.btcHoldings.at(-1)) ? props.btcHoldings.at(-1).value.toFixed(2) : "0.00"}
                    </Typography>
                </Box>
                <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: 4, borderBottom: "1px solid rgba(47,47,47,1)" }}>
                    <Typography style={{ textAlign: "left", whiteSpace: "nowrap" }} aria-label="Bitcoin Position">
                        Bitcoin Position
                    </Typography>
                    <Typography aria-label={(props.btcPosition.at(-1)) ? props.btcPosition.at(-1).value.toFixed(2) : "0.00"}>
                        {(props.btcPosition.at(-1)) ? props.btcPosition.at(-1).value.toFixed(2) : "0.00"}
                    </Typography>
                </Box>
            </Box>
		</>
	)
}

export default PortfolioMetrics