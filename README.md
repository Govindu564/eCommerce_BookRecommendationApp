# E-Commerce Book Recommendation App  

A Full-Stack MERN + Machine Learning project that recommends books based on category, genre, and user preferences.  
It combines a Node/Express backend, a React frontend, and Python scripts for ML-based recommendations.

---

# Tech Stack

## Frontend
- React (Vite)
- Bootstrap / React-Bootstrap
- Axios for API calls

## Backend
- Node.js / Express.js
- MongoDB 
- REST API Architecture

## Machine Learning
- Python 
- Jupyter Notebooks
- Data Cleaning & Recommendation Model

---

# Folder Structure

BookRecommendation/
├── backend/ # Express backend API
│ ├── controllers/
│ ├── models/
│ ├── routers/
│ ├── index.js
│ └── package.json
│
├── react-recommender/ # React frontend (Vite)
│ ├── public/
│ ├── src/
│ ├── package.json
│ └── README.md
│
├── script.py # Python script for recommendation logic
├── code.ipynb # ML notebook
└── cleaned_data.csv # Preprocessed dataset


---

# Setup Instructions

# 1. Clone the Repository
```bash
git clone https://github.com/Govindu564/eCommerce_BookRecommendationApp.git
cd eCommerce_BookRecommendationApp

# 2. Run the Backend
cd backend
npm install
node index.js


# 3. Run the Frontend
cd react-recommender
npm install
npm run dev


4️. Run the ML Scripts
python script.py


# Features

 User Authentication (Login / Signup)

 Cart Management & Book Checkout

 Search & Filter Books by Category / Genre

 ML-Based Book Recommendations

 Responsive UI (Bootstrap)

 Integrated Python ML for Smart Suggestions

 