### **🎬 Hybrid Movie Recommendation System** | **Python, FAISS, Sentence-Transformers, Scikit-learn, Streamlit, Docker, AWS EC2**

* Built a hybrid movie recommendation system that combines **semantic content-based retrieval** using **Sentence-Transformers** and **FAISS** with **collaborative filtering (SVD)** to provide personalized movie recommendations.
* Implemented intelligent recommendation routing based on user interaction history and addressed the **cold-start problem** with a content-based fallback strategy for new users and movies.
* Integrated the **OMDb API** to dynamically fetch and display movie posters and metadata, enhancing the user experience with rich visual recommendations.
* Containerized the application using **Docker** and deployed it on an **AWS EC2** instance with **Streamlit**, securely managing API keys through environment variables and optimizing performance with cached model loading.
* Designed an interactive web interface that delivers fast, scalable, and user-friendly movie recommendations in real time.


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
- FAISS
- Sentence Transformers

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

## 🧠 Machine Learning Workflow

```text
                         Movie Dataset (TMDB)
                                 │
                                 ▼
                  Data Cleaning & Preprocessing
                                 │
                                 ▼
                      Feature Engineering
                 (Genres, Overview, Keywords, Tags)
                                 │
             ┌───────────────────┴───────────────────┐
             ▼                                       ▼
     Content-Based Filtering             Collaborative Filtering
             │                                       │
             ▼                                       ▼
   Sentence-Transformer Embeddings       User-Movie Ratings Matrix
             │                                       │
             ▼                                       ▼
      FAISS Vector Index                 SVD Matrix Factorization
   (Semantic Similarity Search)            (Surprise Library)
             │                                       │
             └───────────────────┬───────────────────┘
                                 ▼
                        Hybrid Recommendation Engine
               (Routes users based on interaction history)
                                 │
         ┌───────────────────────┼────────────────────────┐
         ▼                       ▼                        ▼
     Warm User              Cold/New User           No Interaction
         │                       │                        │
         ▼                       ▼                        ▼
  SVD Recommendations     Content-Based Search    Popularity-Based Results
         │                       │                        │
         └───────────────────────┼────────────────────────┘
                                 ▼
                  OMDb API (Movie Posters & Metadata)
                                 │
                                 ▼
                    Streamlit Web Application (UI)
                                 │
                                 ▼
                    Docker Container → AWS EC2 Instance
```

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

- GitHub: https://github.com/Aryannchoudhary
- LinkedIn: https://linkedin.com/in/aryan176


---

## ⭐ If you found this project helpful, consider giving it a star!