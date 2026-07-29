# 🎬 Movie Recommendation System

A content-based Movie Recommendation System built using Machine Learning that recommends similar movies based on user selection. The application is deployed on AWS EC2 using Docker and displays movie posters using the OMDb API.

---

## 🚀 Live Demo

**Application:** http://32.236.115.18:8501

---



## ✨ Features

- 🎥 Content-Based Movie Recommendation
- 🔍 Search from thousands of movies
- 🖼️ Dynamic movie posters using OMDb API
- ⚡ Fast recommendation generation
- 🐳 Dockerized application
- ☁️ Deployed on AWS EC2
- 🎨 Interactive Streamlit UI

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Pandas
- NumPy

### Visualization

- Streamlit

### API

- OMDb API

### Deployment

- Docker
- AWS EC2
- Linux (Ubuntu)

---

## 📂 Project Structure

```
Movie_Recommender_System/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── movies.pkl
├── similarity.pkl
├── README.md
├── screenshots/
└── assets/
```

---

## ⚙️ How It Works

1. User selects a movie.
2. The system finds similar movies using cosine similarity.
3. Recommended movie titles are generated.
4. OMDb API fetches movie posters.
5. Results are displayed in the Streamlit interface.

---

## 🧠 Machine Learning Workflow

```
Movie Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Vectorization
      │
      ▼
Cosine Similarity Matrix
      │
      ▼
Recommendation Engine
      │
      ▼
Streamlit Application
```

---

## ☁️ Deployment Architecture

```
                    GitHub
                       │
                       ▼
                 Docker Image
                       │
                       ▼
                AWS EC2 Instance
                       │
                       ▼
                Docker Container
                       │
                       ▼
             Streamlit Application
                       │
                       ▼
                 OMDb Movie API
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t movie-recommender .
```

### Run Container

```bash
docker run -d \
--name movie-app \
-p 8501:8501 \
-e OMDB_API_KEY=YOUR_API_KEY \
movie-recommender
```

---

## 💻 Local Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Movie_Recommender_System.git
```

Move into the project

```bash
cd Movie_Recommender_System
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

Create an environment variable:

```
OMDB_API_KEY=YOUR_API_KEY
```

---

## 📈 Future Improvements

- User authentication
- Movie ratings integration
- Genre-based filtering
- Trending movies
- TMDB integration
- Personalized recommendations
- Custom domain deployment

---

## 👨‍💻 Author

**Aryan Choudhary**

- GitHub: https://github.com/YOUR_USERNAME
- LinkedIn: https://linkedin.com/in/YOUR_PROFILE

---

## ⭐ If you found this project helpful, consider giving it a star!