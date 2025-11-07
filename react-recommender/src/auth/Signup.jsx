import React, { useState } from "react";
import axios from "axios";
import { useAuth } from "../authContext";
import { Button } from "@primer/react";
import { Link } from "react-router-dom";
import "./Login.css";

const Signup = () => {
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const { setCurrentUser } = useAuth();

  const validateInputs = () => {
    if (!email || !username || !password) {
      setError("All fields are required");
      return false;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setError("Invalid email format");
      return false;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return false;
    }
    setError("");
    return true;
  };
  const handleSignup = async (e) => {
    e.preventDefault();
    if (!validateInputs()) return;

    try {
      setLoading(true);
      const res = await axios.post("http://localhost:3002/signup", {
        email,
        password,
        username,
      });

      setLoading(false);
      window.location.href = "/auth";
    } catch (err) {
      console.error("Signup Error:", err);
      setError("Signup failed. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h2 className="login-title"> Create Account</h2>

        {error && <p className="error-message">{error}</p>}

        <form onSubmit={handleSignup} className="login-form">
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              placeholder="Enter your username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label>Email address</label>
            <input
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="form-input"
            />
          </div>

          <Button
            variant="primary"
            type="submit"
            className="login-button"
            disabled={loading}
          >
            {loading ? "Signing Up..." : "Sign Up"}
          </Button>
        </form>

        <p className="signup-text">
          Already have an account? <Link to="/auth">Login</Link>
        </p>
      </div>
    </div>
  );
};

export default Signup;
