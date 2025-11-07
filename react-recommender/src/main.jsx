import { createRoot } from "react-dom/client";
// import "./index.css";
import { AuthProvider } from "./authContext.jsx";
import { BrowserRouter, Routes, Route, Router } from "react-router-dom";
import ProjectRoutes from "./routes.jsx";

createRoot(document.getElementById("root")).render(
  <AuthProvider>
    <BrowserRouter>
      <ProjectRoutes />
    </BrowserRouter>
  </AuthProvider>
);