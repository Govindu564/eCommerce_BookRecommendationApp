import React, { useEffect } from "react";

import { useNavigate, useRoutes } from "react-router-dom";

import Signup from "./auth/Signup";
import Login from "./auth/Login";
import App from "./App";

import { useAuth } from "./authContext";

const ProjectRoutes = () => {
  const { currentUser, setCurrentUser } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const userIdfromStorage = localStorage.getItem("userId");
    if (userIdfromStorage && !currentUser) {
      setCurrentUser(userIdfromStorage);
    }

    if (
      !userIdfromStorage &&
      !["/auth", "/signup"].includes(window.location.pathname)
    ) {
      navigate("/auth");
    }

    if (userIdfromStorage && window.location.pathname == "/auth") {
      navigate("/");
    }
  }, [currentUser, navigate, setCurrentUser]);

  let element = useRoutes([
    {
      path: "/",
      element: <App />,
    },
    {
      path: "/auth",
      element: <Login />,
    },
    {
      path: "/signup",
      element: <Signup />,
    },
    // {
    //     path:"/footer",
    //     element:<Footer/>
    // },
  ]);
  return element;
};

export default ProjectRoutes;
