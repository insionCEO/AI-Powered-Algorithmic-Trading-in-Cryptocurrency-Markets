import React from 'react';
import { Button } from '@mui/material';
import Selector from './Selector';

const DatasetConnect = (props) => {

  	return (
		<div style={{ display: "flex", justifyContent: "space-around", alignItems: "center", height: "200" }}>
            <Selector
            options={props.datasets}
            label="Select Dataset"
            value={props.dataset}
            onChange={props.onDatasetChange}
            width={130}
            />
            <Button variant="outlined" color="success" onClick={props.onConnect}>
                Connect
            </Button>
        </div>
  	);
};

export default DatasetConnect;
