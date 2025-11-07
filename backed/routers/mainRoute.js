const express = require("express");
const userRouter = require("./userRoute");

const mainRouter = express.Router();

mainRouter.use(userRouter);

mainRouter.get("/", (req, res) => {
  res.send("welcome main route!");
});
module.exports = mainRouter;
