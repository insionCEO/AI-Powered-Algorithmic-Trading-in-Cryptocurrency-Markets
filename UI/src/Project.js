import React from 'react'
import { Typography, Grid, Paper, Box, List, ListItem, ListItemText } from '@mui/material';
import ScrollToTop from './ScrollToTopButton';

import 'katex/dist/katex.min.css';
import Latex from 'react-latex-next';

const Project = () => {
	return (
		<Grid container sx={{ p: 2, flexDirection: "row", display: "flex", overflowX: "hidden", overflowY: "hidden" }}>
			<Grid item xs={3} sx={{ display: "flex", height: "100%", p: 2, paddingTop: 10, alignItems: "center" }}>
				<div>
					<Typography variant="h4" sx={{ textDecoration: "underline" }}>Contents</Typography>
					<List>
						<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#DataMethods">
							<ListItemText primary="1. Data Methods" />
						</ListItem>
						<List sx={{ paddingLeft: 4, paddingTop: 0, paddingBottom: 0 }}>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#CollectionAndStorage">
								<ListItemText primary="1.1 Collection and Storage" />
							</ListItem>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#FeatureExtraction">
								<ListItemText primary="1.2 Feature Extraction" />
							</ListItem>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#DataEvaluation">
								<ListItemText primary="1.3 Evaluation" />
							</ListItem>
						</List>
						<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#PredictiveModels">
							<ListItemText primary="2. Predictive Models" />
						</ListItem>
						<List sx={{ paddingLeft: 4, paddingTop: 0, paddingBottom: 0 }}>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#DataPreparation">
								<ListItemText primary="2.1 Data Preparation" />
							</ListItem>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#Transformers">
								<ListItemText primary="2.2 Transformers" />
							</ListItem>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#Informer">
								<ListItemText primary="2.3 Informer" />
							</ListItem>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#Pyraformer">
								<ListItemText primary="2.4 Pyraformer" />
							</ListItem>
						</List>
						<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#TradingStrategies">
							<ListItemText primary="3. Trading Strategies" />
						</ListItem>
						<List sx={{ paddingLeft: 4, paddingTop: 0, paddingBottom: 0 }}>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#RL">
								<ListItemText primary="3.1 RL-Based Strategies" />
							</ListItem>
							<ListItem button sx={{ paddingTop: 0, paddingBottom: 0 }} component="a" href="#HeuristicStrategies">
								<ListItemText primary="3.2 Heuristic Strategies" />
							</ListItem>
						</List>
					</List>
				</div>
			</Grid>
			<Grid item xs={9} sx={{ p: 2, borderLeft: "1px solid rgba(47,47,47,1)" }}>
				<Grid item>
					<Typography variant='h2' xs={12} gutterBottom id="DataMethods">Data Methods</Typography>
					<Typography variant='h4' id="CollectionAndStorage">Collection and Storage</Typography>
					<Latex>
						Limit order book data of cryptocurrency assets is gathered through the Binance exchange market with the use of their API service that offers latest and historic data, additionally, limit order book data, that is already transformed, is also collected from Kaggle for testing purposes. The dataset collected from Binance represents 7 days of Bitcoin trading activity where, after feature extraction, the dataset will be split in different configurations for training and testing of the machine learning models. Naturally, the data received from the exchange platform API requires a significant amount of storage space. Under these circumstances a data versioning control software tool such as DVC assists with keeping the data separate from the project’s repository by keeping a hash value corresponding to a data file within the project while storing the actual data in the cloud, thus, allowing for the project's repository to be more compact. 
					</Latex>
					<Typography variant='h4' id="FeatureExtraction">Feature Extraction</Typography>
					<Latex>
						After a request is sent to the API, containing the symbol of a cryptocurrency asset and the time period, a response is received that gives access to limit order book data in the format that is displayed in table in Section 2. There are multiple entries for every timestamp that capture all the buy and sell orders at that particular time. Since the limit order book represents what were once real time events the difference between consecutive timestamps is only several milliseconds. To address that, starting from the first time field, timestamp ranges are computed that bundle together consecutive timestamps where each timestamp range represents 1 second of information. After this operation, the newly computed time ranges would be traversed iteratively and at that stage feature extraction can be done on every iteration for the corresponding 1 second range. At every step the lowest ask price and highest bid price are considered the 1-st level of the order book information that resides in the particular time range, the second lowest ask and highest bid would be level 2, the third - level 3, etc. Leveraging the literature review and considering the complexity of storing long property vectors, the features presented in Table are extracted from the limit order book information that resides in each timestamp range, where P denotes Price and V denotes Volume. 
					</Latex>
					<Typography variant='h4' id="DataEvaluation">Evaluation</Typography>
					<Typography paragraph>To evaluate the correctness of the developed data transformation software that converts the limit order book into features, a historic dataset is extracted from Kaggle that contains the midpoint feature while, for the same period of time, data in limit order book format is extracted from Binance and transformed. The resulting graph of the midpoint values from the transformation is compared side to side to the Kaggle midpoint data and are presented in Figure. The comparison confirms that the developed by the team feature extraction script computes accurately the trends within the data.</Typography>
				</Grid>
				<Grid item>
					<Typography variant='h2' id="PredictiveModels">Predicitve Models</Typography>
					<Typography variant='h4' id="DataPreparation">Data Preparation</Typography>
					<Latex>
						Various pre-processing techniques were applied to prepare the processed order book data for training. The initial pre-processing step involved converting the UNIX timestamps column to date/time and transforming the date column into more informative time features, such as month, day, weekday, hour, minute, and second. By incorporating these time features into the model, the model can better account for seasonality, trends, and cyclical patterns typically present in time series data. This improves the model's ability to accurately capture patterns and generate more precise predictions. This process has been previously validated [reference for this].

						The subsequent step in pre-processing entails adjusting the input data structure and shape based on whether the data is multi-variate or single-variate. For multivariate data, multiple features are used, while for single-variate data, only the target column is chosen. The resulting data frame contains columns in the form of $['date', ...(other features), target feature]$ for multivariate data and $['date', target feature]$ for single-variate data.

						Lastly, normalization ensures all feature values are on a similar scale. This is important as the raw data can often have uneven scales, which may cause some features to dominate during training. Normalization also speeds up the training process without compromising performance by guaranteeing that the gradients propagated during backpropagation are more stable and can be calculated more efficiently.
					</Latex>
					<Typography variant='h4' id="Transformers">Transformers</Typography>
					<Latex>
						The Transformer architecture consists of two key components: the Encoder and the Decoder. 

						Encoder is responsible for generating hidden representations of an input sequence. These representations summarize the meaning of each token in the context of the entire sequence and are subsequently fed as input to the Decoder module. The Encoder module comprises a stack of identical layers, each containing two sub-layers: a multi-head self-attention mechanism and a fully connected feed-forward network. The multi-head self-attention mechanism enables the Encoder to capture global dependencies between the elements of an input sequence. At the same time, the feed-forward network introduces nonlinearity as the output from the multi-head attention mechanism is linear.

						Decoder module of the Transformer architecture generates an output sequence based on the hidden representations generated by the Encoder module. Like the Encoder, the Decoder module comprises a stack of identical layers, each with two sub-layers from the Encoder module and an additional third sub-layer performing multi-head attention over the output of the Encoder stack. The self-attention sub-layer in the decoder stack is modified to prevent positions from attending to subsequent positions. The output embeddings are offset by one position to ensure that the predictions for position $i$ depend only on the known outputs at positions less than $i$.
					</Latex>
					<Typography variant='h4' id="Informer">Informer</Typography>
					<Latex>
						Informer is a modified version of the Transformer architecture incorporating three key changes: the ProbSparse self-attention mechanism, the self-attention distilling technique, and the generative-style decoder. These modifications enable Informer to achieve a time and memory complexity of $O(N \log N)$, substantially improving the original Transformer architecture. Furthermore, the number of decoder operations is reduced from $L$ to $O(1)$, where $L$ represents the sequence length. Specifically, the ProbSparse self-attention mechanism and self-attention distilling technique allow for efficient computation of attention weights. At the same time, the generative-style decoder provides a more effective approach for generating output sequences.
					</Latex>
					<Typography variant='h4' id="Pyraformer">Pyraformer</Typography>
					<Latex>
						Pyraformer proposes a novel pyramidal attention-based Transformer that aims to capture long-range dependencies in time series data while maintaining low time and space complexity. The authors of Pyraformer state that calculating dependencies between different data points in a sequence can be modeled on a graph, where shorter paths between points lead to better dependency capture. The self-attention module has a max path of $O(1)$ as each point attended to all the points at the cost of quadratic time complexity. In contrast, RNN and CNN models typically have a max path of $O(L)$ but have a time complexity of $O(L)$.
					</Latex>
				</Grid>
				<Grid item> 
					<Typography variant='h2' id="TradingStrategies">Trading Strategies</Typography>
					<Typography variant='h4' id="RL">RL-Based Strategies</Typography>
					<Latex>
						Although complex, it was prevalent in the literature that several Reinforcement Learning strategies could be beneficial for our application. Still, few research papers explored the usage of this technology within the Cryptocurrency market, and none were found that utilised data from predictive models as part of their environment definition. The following section of this report describes how RL was incorporated into this project and what decisions in architecture were made. Many of the approaches discussed are either quite novel or experimental.
					</Latex>
					<Typography variant='h4' id="HeuristicStrategies">Heuristic-Based Strategies</Typography>
					<Latex>
						After a request is sent to the API, containing the symbol of a cryptocurrency asset and the time period, a response is received that gives access to limit order book data in the format that is displayed in table in Section 2. There are multiple entries for every timestamp that capture all the buy and sell orders at that particular time. Since the limit order book represents what were once real time events the difference between consecutive timestamps is only several milliseconds. To address that, starting from the first time field, timestamp ranges are computed that bundle together consecutive timestamps where each timestamp range represents 1 second of information. After this operation, the newly computed time ranges would be traversed iteratively and at that stage feature extraction can be done on every iteration for the corresponding 1 second range. At every step the lowest ask price and highest bid price are considered the 1-st level of the order book information that resides in the particular time range, the second lowest ask and highest bid would be level 2, the third - level 3, etc. Leveraging the literature review and considering the complexity of storing long property vectors, the features presented in Table are extracted from the limit order book information that resides in each timestamp range, where denotes Price and denotes Volume.
					</Latex>
				</Grid>
			</Grid>
			<ScrollToTop />
		</Grid>
	)
}

export default Project