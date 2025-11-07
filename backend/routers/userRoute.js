const express = require("express");
const  userController = require("../controllers/userController");

const userRouter = express.Router();

userRouter.post("/signup",userController.signup);
userRouter.post("/login",userController.login);
userRouter.get("/allusers",userController.getAllUsers);
userRouter.get("/userprofile/:id",userController.getUserProfile);
userRouter.post("/updateuser/:id",userController.updateUser);
userRouter.get("/deleteuser/:id",userController.deleteUser);

module.exports  = userRouter;