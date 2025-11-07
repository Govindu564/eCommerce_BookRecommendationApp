const express = require("express");
const app = express();
const PORT = 3002;
const bodyParser = require('body-parser');
const cors = require('cors');
const dotenv = require("dotenv");
const mongoose = require('mongoose');
const mainRouter = require("./routers/mainRoute");
const path = require('path');


dotenv .config();


app.use(bodyParser.json());
app.use(cors());
app.use(bodyParser.urlencoded({ extended: true })); 
app.use(mainRouter);


mongoose
  .connect(process.env.MONGODB_URL, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(() => console.log('MongoDB connected successfully!'))
  .catch((err) => console.error('Error connecting to MongoDB:', err));

app.get("/",(req,res)=>{
    res.send("this is root of youtube");
});

app.listen(PORT,(req,res)=>{
    console.log(`This port is listening :${PORT}`);
});