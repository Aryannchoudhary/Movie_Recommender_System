# 🎬 Hybrid Movie Recommendation System

An end-to-end **Hybrid Movie Recommendation System** that combines **Content-Based Filtering** and **Collaborative Filtering** to deliver personalized movie recommendations. The application intelligently switches between recommendation strategies based on user interaction history, solving the cold-start problem while maintaining recommendation quality.

The project is fully **Dockerized**, **deployed on AWS EC2**, and integrated with **GitHub Actions CI/CD** for automatic deployments.

---

## 🚀 Live Demo

**Application:** http://32.236.115.18:8501

---

## ✨ Features

- 🎯 Hybrid recommendation engine (Content-Based + Collaborative Filtering)
- 🔍 Semantic movie search using Sentence Transformers
- ⚡ Fast similarity search with FAISS
- ⭐ Personalized recommendations using SVD Matrix Factorization
- 🆕 Cold-start handling for new users
- 🎭 Genre-based fallback recommendations
- 📊 Popularity-based recommendations when no user history exists
- 🖼️ Movie posters and metadata using the OMDb API
- 💾 Cached model loading for faster performance
- 🐳 Dockerized application for portability
- ☁️ Deployed on AWS EC2
- 🔄 Automated CI/CD pipeline with GitHub Actions

---

# 🏗️ Project Architecture

```
                        User
                          │
                          ▼
                 Streamlit Web App
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
 Content-Based Engine          Collaborative Engine
         │                                 │
 Sentence Transformers             Surprise (SVD)
         │                                 │
      FAISS Index                  User Ratings
         │                                 │
         └───────────────┬─────────────────┘
                         ▼
                 Hybrid Recommendation
                         │
                         ▼
                OMDb API Integration
                  (Movie Posters)
```

---

# 🧠 Machine Learning Workflow

```
                              Movie Dataset (TMDB)
                                      │
                                      ▼
                              Data Cleaning &
                            Deduplication
                                      │
                                      ▼
                            Feature Engineering
                          (genres, overview, tags)
                                      │
                ┌─────────────────────┴─────────────────────┐
                ▼                                             ▼
      Content-Based Path                          Collaborative Path
                │                                             │
                ▼                                             ▼
      Sentence-Transformer                         User-Item Interactions
         Embeddings                                (ratings / watch history)
                │                                             │
                ▼                                             ▼
        FAISS Vector Index                          Train/Test Split
      (semantic similarity                       (warm users vs. held-out
         search)                                   cold users & cold items)
                │                                             │
                │                                             ▼
                │                                    SVD Matrix Factorization
                │                                        (Surprise)
                │                                             │
                └─────────────────────┬─────────────────────┘
                                      ▼
                          Hybrid Router
                (Warm User / Cold User Detection)
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
        Warm User              Cold User              No User History
             │                      │                      │
             ▼                      ▼                      ▼
 Collaborative Filtering     Content-Based        Popularity-Based
  Recommendations           Recommendations      Recommendations
                │                     │                     │
                └─────────────────────┼─────────────────────┘
                                      ▼
                           Streamlit Web Application
```

---

# ☁️ Deployment Architecture

```
                   ┌───────────────────┐
                   │     Developer     │
                   └─────────┬─────────┘
                             │
                      git push origin main
                             │
                             ▼
                   ┌───────────────────┐
                   │ GitHub Repository │
                   └─────────┬─────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │ GitHub Actions (CI/CD) │
                 └─────────┬──────────────┘
                           │
                     SSH Deployment
                           │
                           ▼
                 ┌────────────────────────┐
                 │ AWS EC2 (Ubuntu)       │
                 └─────────┬──────────────┘
                           │
                    git pull origin main
                           │
                    docker build
                           │
               Stop Old Docker Container
                           │
               Start New Docker Container
                           │
                           ▼
                 ┌────────────────────────┐
                 │ Streamlit Application  │
                 │      Port : 8501       │
                 └─────────┬──────────────┘
                           │
                           ▼
                 ┌────────────────────────┐
                 │      OMDb API          │
                 │ Posters & Metadata     │
                 └────────────────────────┘
```

---

# 🔄 CI/CD Pipeline

The project uses **GitHub Actions** to automate deployments. Every push to the `main` branch automatically updates the application running on the AWS EC2 instance.

## Workflow

```
Developer
    │
    ▼
git push origin main
    │
    ▼
GitHub Repository
    │
    ▼
GitHub Actions
    │
    ▼
SSH into AWS EC2
    │
    ▼
Pull Latest Code
    │
    ▼
Build Docker Image
    │
    ▼
Stop Existing Container
    │
    ▼
Start New Container
    │
    ▼
Updated Streamlit Application
```

### CI/CD Features

- Automatic deployment on every push to `main`
- Secure SSH authentication using GitHub Secrets
- Automatic Docker image rebuild
- Automatic container restart
- No manual deployment required
- Secure environment variable management using `.env`

---

# 📂 Project Structure

```
Movie_Recommender_System/
│
├── models/
│   ├── faiss_index.bin
│   ├── sentence_embeddings.pkl
│   ├── svd_model.pkl
│   └── ...
│
├── src/
│   ├── recommender.py
│   ├── preprocessing.py
│   ├── utils.py
│   └── ...
│
├── screenshots/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
└── .github/
    └── workflows/
        └── deploy.yml
```

---

# 🛠️ Tech Stack

### Machine Learning

- Python
- Scikit-learn
- Sentence Transformers
- FAISS
- Surprise (SVD)

### Web Application

- Streamlit

### APIs

- OMDb API

### Deployment

- Docker
- AWS EC2
- GitHub Actions

### Version Control

- Git
- GitHub

---

# 📊 Recommendation Strategy

| Scenario | Recommendation Method |
|----------|------------------------|
| Existing User | Collaborative Filtering (SVD) |
| New User | Content-Based Filtering |
| New Movie | Semantic Similarity Search |
| No User History | Popularity-Based Recommendations |

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/<your-username>/<repository>.git
cd Movie_Recommender_System
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Create Environment File

```env
OMDB_API_KEY=YOUR_OMDB_API_KEY
```

## Run Application

```bash
streamlit run app.py
```

---

# 🐳 Docker

## Build Image

```bash
docker build -t movie-recommender .
```

## Run Container

```bash
docker run -d \
--name movie-app \
-p 8501:8501 \
--env-file .env \
movie-recommender
```

---


# 📈 Future Improvements

- User authentication
- User profiles
- Watchlist functionality
- Movie trailer integration
- Multi-language support
- Kubernetes deployment
- Monitoring with Prometheus & Grafana
- HTTPS with Nginx and Let's Encrypt

---

# 👨‍💻 Author

**Aryan Choudhary**

- GitHub: https://github.com/Aryannchoudhary
- LinkedIn: https://linkedin.com/in/aryan176

---

## ⭐ If you found this project useful, consider giving it a star!