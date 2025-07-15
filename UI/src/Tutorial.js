import React, { useState } from 'react';
import { Card, CardContent, Typography, Dialog, IconButton, Box } from '@mui/material';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import CloseIcon from '@mui/icons-material/Close';

import empty_full_page from './assets/Tutorial/empty_full_page.png'
import empty_portfolio from './assets/Tutorial/empty_portfolio.png'
import strategy_select_dropdown from './assets/Tutorial/strategy_select_dropdown.png'
import default_strategy_section from './assets/Tutorial/default_strategy_section.png'
import empty_strategy_history from './assets/Tutorial/empty_strategy_history.png'
import empty_strategy_history_hovered from './assets/Tutorial/empty_strategy_history_hovered.png'
import populated_full_page from './assets/Tutorial/populated_full_page.png'
import populated_strategy_table from './assets/Tutorial/populated_strategy_table.png'
import portfolio_dataset_connect_dropdown from './assets/Tutorial/portfolio_dataset_connect_dropdown.png'
import populated_bitcoin_graph from './assets/Tutorial/populated_bitcoin_graph.png'
import populated_portfolio from './assets/Tutorial/populated_portfolio.png'
import project_highlighted from './assets/Tutorial/project_highlighted.png'

const Tutorial = ({ isOpen, onClose }) => {
	const [currentCard, setCurrentCard] = useState(0);
	const cards = [
		{
			title: 'Welcome to TradeAI!',
			content: 'This app is designed to give an interactive experience with our AI powered trading system.',
			
		},
		{
			title: 'At first glance.',
			content: 'Initially the page will look sparse. That is because you\'ve not connected to the server.',
			imageSrc: empty_full_page,
			alt: "Empty full page"
		},
		{
			title: 'Connecting to the server.',
			content: 'You may have to scroll down the page but there should be a section which allows for the selection of dataset as well as a connect button. Once you\'ve chosen a dataset, press connect and the page should fill with graphs and information.',
			imageSrc: portfolio_dataset_connect_dropdown,
			alt: "Portfolio dataset connect dropdown",
		},
		{
			title: 'You\'re now connected.',
			content: 'Once connected your page should start to look a little like this.',
			imageSrc: populated_full_page,
			alt: "Populated full page"
		},
		{
			title: 'Price Graph',
			content: 'The top graph displays the price of Bitcoin to the US Dollar, as well as the predicted price from our back-end. You are able to interact with the graph by: Scrolling to zoom, Dragging to move around the graph as well as you are able to click and drag the prices shown on the right of the graph to increase the granularity or get a bigger picture.',
			imageSrc: populated_bitcoin_graph,
			alt: "Populated full page"
		},
		{
			title: 'Portfolio',
			content: 'Here lies the information regarding the current portfolio status. You can find the total cash value being held, the bitcoin amount being held and more. The values are nicely displayed on the left whilst also being updated in the graphs. Again these graphs are interactable the same as the price graph.',
			imageSrc: populated_portfolio,
			alt: "Populated portfolio"
		},
		{
			title: 'Strategy Selection',
			content: 'At the bottom section of the page is the strategy area. This is where you can configure the AI model used for prediction and the trading methodology. Once selected you can press Execute and the server should receive the configuration. Additionally if you would like to stop the strategy the Stop button is there for that purpose.',
			imageSrc: strategy_select_dropdown,
			alt: "Strategy select model dropdown"
		},
		{
			title: 'Strategy History',
			content: 'Underneath lies the Strategy History which (when strategies are ran) will allow the user to visualise the fill orders executed by the back end trading strategy.',
			imageSrc: empty_strategy_history,
			alt: "Empty strategy history"
		},
		{
			title: 'Strategy History continued',
			content: 'Once fill orders are visualised on the price graph at the top, or when the trading strategy has run its course and no new values are being updated, Press the Stop button and that run will now be visible in the strategy history.',
			imageSrc: empty_strategy_history_hovered,
			alt: "Empty strategy history hovered"
		},
		{
			title: 'Strategy History Data',
			content: 'When clicked on the section will populate with a table providing all the information received from the server relating to the strategy and the orders executed.',
			imageSrc: populated_strategy_table,
			alt: "Populated strategy table"
		},
		{
			title: 'Project Information',
			content: 'For more information regarding the work that went into the project, and the back end please refer to the project section which can be accessed from the navbar at the top.',
			imageSrc: project_highlighted,
			alt: "Project highlighted"
		},
		{
			title: 'Project Information',
			content: 'For more information regarding the work that went into the project, and the back end please refer to the project section which can be accessed from the navbar at the top.',
			imageSrc: project_highlighted,
			alt: "Project highlighted"
		},
	];

	const handleNext = () => {
		if (currentCard < cards.length - 1) {
			setCurrentCard(currentCard + 1);
		}
	};

	const handleBack = () => {
		if (currentCard > 0) {
			setCurrentCard(currentCard - 1);
		}
	};

	return (
		<Dialog open={isOpen} onClose={onClose} fullWidth maxWidth="md" >
			<div
			style={{
				display: 'flex',
				alignItems: 'center',
				justifyContent: 'center',
				height: '100%',
				position: 'relative',
			}}
			>
			<div
				style={{
				position: 'absolute',
				top: 0,
				left: 0,
				bottom: 0,
				right: 0,
				backgroundColor: 'rgba(0, 0, 0, 0.5)',
				zIndex: -1,
				}}
				onClick={onClose}
			/>
				<Card style={{ width: "100%" }}>
					<CardContent sx={{ textAlign: "center" }}>
						<div style={{ display: 'flex', justifyContent: 'flex-end' }}>
							<IconButton onClick={onClose}>
								<CloseIcon />
							</IconButton>
						</div>
						<div style={{ display: 'flex', justifyContent: 'space-between', alignItems: "center", height: "100%" }}>
							<ChevronLeftIcon
							onClick={handleBack}
							disabled={currentCard === 0}
							sx={{ cursor: 'pointer' }}
							/>
							<Box p={2}>
								<Typography variant="h5" component="div" sx={{ mb: 2 }} p={2}>
									{cards[currentCard].title}
								</Typography>
								{cards[currentCard].imageSrc && (<img src={cards[currentCard]?.imageSrc} alt={cards[currentCard]?.alt} width={500} />)}
								<Typography sx={{ mb: 1.5, p: 2 }} color="text.secondary">
									{cards[currentCard].content}
								</Typography>
								<Typography sx={{ mb: 1.5 }} color="text.secondary">
									Card {currentCard + 1} of {cards.length}
								</Typography>
							</Box>
							<ChevronRightIcon
							onClick={handleNext}
							disabled={currentCard === cards.length - 1}
							sx={{ cursor: 'pointer' }}
							/>
						</div>
					</CardContent>
				</Card>
			</div>
		</Dialog>
	);
};

export default Tutorial;
