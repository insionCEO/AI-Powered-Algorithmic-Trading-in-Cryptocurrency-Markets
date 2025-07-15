import React from "react";
import { useState } from "react";
import { Typography, AppBar, Toolbar, CssBaseline, MenuItem, Box, Button, Paper, Container } from '@mui/material';

import Dashboard from "./Dashboard";
import Project from "./Project";

import TradeAILogo from './assets/images/tradeai_logo_symbol.png';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import { ThemeProvider, createTheme } from "@mui/material/styles";
import Tutorial from "./Tutorial";

const darkTheme = createTheme({
	palette: {
		mode: 'dark',
	},
});


const App = () => {
	
	const pages = ['Trading', 'Project'];
	const [showTutorial, setShowTutorial] = useState(true);
	
	const handleTutorialClose = () => {
		setShowTutorial(false);
	}

	const [currentPage, setCurrentPage] = useState('Trading');

	const handlePageChange = (event) => {
		console.log(event.target.textContent);
		const newPage = event.target.textContent;
		setCurrentPage(newPage);
	}

	return (
		<>
			<ThemeProvider theme={darkTheme} >
				<CssBaseline />
					<AppBar position="static" role="banner">
						<Toolbar sx={{ display: "flex", justifyContent: "space-around", alignItems: "center" }}>
							<Box sx={{ flex: 1, display: "flex", justifyContent: "center", alignItems: "center" }}>
								<img src={TradeAILogo} height="48" xs="auto" alt="TradeAILogo" aria-label="TradeAI Logo"/>
								<Typography onClick={() => {setCurrentPage('Trading')}} variant="h4" xs={6} sx={{ flex: 1}}>
									TradeAI
								</Typography>
							</Box>
							<Box sx={{ flex: 1, display: "flex", justifyContent: "flex-end", alignItems: "center" }}>
							{ pages.map((page) => (
								<MenuItem tabIndex={0} onClick={handlePageChange} aria-label={page}>
									<Typography textAlign="center">{page}</Typography>
								</MenuItem>
							))}
							</Box>
							<Box sx={{ display: "flex", justifyContent: "center", alignItems: "center", p: 2}}>
								<HelpOutlineIcon onClick={() => {setShowTutorial(true)}} sx={{ cursor: "pointer", transition: "all 0.3s", "&:hover": { transform: "scale(1.1)" }, }} />
							</Box>
						</Toolbar>
					</AppBar>
				<Container maxWidth="xl" >
					<Box>
						{ showTutorial === true && <Tutorial isOpen={showTutorial} onClose={handleTutorialClose} />}
						{/* Render the current page based on the state */}
						{currentPage === 'Trading' && <Dashboard />}
						{currentPage === 'Project' && <Project />}
					</Box>
				</Container>
			</ThemeProvider>
		</>
	);
}

export default App;