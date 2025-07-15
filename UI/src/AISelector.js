import { FormControl, InputLabel, MenuItem, Select } from "@mui/material";
import React from "react";
import { useState } from "react";
import { ChangeEvent } from "react";

const AISelector = () => {

    const [AI, setAI] = useState([
        {
            name: "LSTM",
            time: [
                "1s", "5s", "30s", "1m",
            ]
        },
        {
            name: "Transformers",
            time: [
                "1s", "5s", "30s", "1m",
            ]
        },
        {
            name: "Deep Reinforcement Learning",
            time: [
                "1s", "5s", "30s", "1m",
            ]
        }
    ])

    const handleChange = (event: ChangeEvent<{value: dataType[]}>) => {
        console.log(event.target.value);
        const elem = document.getElementById("select");
        elem.value = event.target.value;
    }

    return (
        <>
            <FormControl fullWidth>
                <InputLabel id="select-ai">
                    AI
                </InputLabel>
                <Select
                labelId="select-ai"
                id="select"
                value={AI.at(0).name}
                label="AI"
                onChange={handleChange}>
                    <MenuItem value={AI.at(0).name}>{AI.at(0).name}</MenuItem>
                    <MenuItem value={AI.at(1).name}>{AI.at(1).name}</MenuItem>
                    <MenuItem value={AI.at(2).name}>{AI.at(2).name}</MenuItem>
                </Select>
            </FormControl>
        </>
    );

};

export default AISelector;