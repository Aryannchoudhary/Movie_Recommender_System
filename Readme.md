# 🎬 Hybrid Movie Recommendation System

A production-ready **Hybrid Movie Recommendation System** built with **Python**, **Streamlit**, **SentenceTransformers**, **FAISS**, and **Collaborative Filtering (Surprise SVD)**. The application combines content-based and collaborative recommendation techniques to provide personalized movie suggestions with rich movie information fetched from the **OMDb API**.

---

## 🚀 Features

- 🎥 Content-Based Movie Recommendations
- 👥 Collaborative Filtering using Surprise SVD
- 🔀 Hybrid Recommendation Engine
- 🔍 Semantic Search using SentenceTransformer
- ⚡ Fast Similarity Search with FAISS
- 🖼 Movie Posters & Details via OMDb API
- 🎨 Interactive Streamlit Interface
- 🐳 Dockerized Application
- ☁️ AWS EC2 Deployment Ready

---

# 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| Recommendation Engine | Hybrid (Content + Collaborative) |
| Embeddings | SentenceTransformer |
| Vector Search | FAISS |
| Collaborative Filtering | Surprise SVD |
| Movie Metadata | OMDb API |
| Containerization | Docker |
| Deployment | AWS EC2 |
| Version Control | Git & GitHub |

---

# 📂 Project Structure

```
Movie_Recommender_System/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── .env
│
├── models/
│   ├── movies.pkl
│   ├── movie_index.faiss
│   ├── indices.pkl
│   ├── cf_model.pkl
│
├── src/
│   ├── collaborative.py
│   ├── content_based.py
│   ├── recommender.py
│   ├── loader.py
│   ├── omdb.py
│   ├── config.py
│   └── utils.py
│
└── notebooks/
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Movie_Recommender_System.git

cd Movie_Recommender_System
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root.

```env
OMDB_API_KEY=your_api_key
```

Get your free API key from:

https://www.omdbapi.com/apikey.aspx

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will be available at

```
http://localhost:8501
```

---

# 🐳 Run with Docker

Build the Docker image

```bash
docker build -t movie-recommender .
```

Run the container

```bash
docker run -p 8501:8501 movie-recommender
```

Open

```
http://localhost:8501
```

---

# ☁️ AWS EC2 Deployment

1. Launch an Ubuntu EC2 instance.
2. Install Docker.
3. Clone the repository.
4. Build the Docker image.

```bash
docker build -t movie-recommender .
```

5. Run the container.

```bash
docker run -d -p 8501:8501 movie-recommender
```

6. Open

```
http://EC2_PUBLIC_IP:8501
```

---

# 🎯 Recommendation Strategies

### 🎬 Content-Based Filtering

- Uses SentenceTransformer embeddings.
- Retrieves similar movies using FAISS cosine similarity.
- Best for users searching by movie title.

---

### 👥 Collaborative Filtering

- Uses Surprise SVD.
- Predicts ratings based on user behavior.
- Best for existing users.

---

### 🎭 Genre-Based Recommendation

- Filters movies by genre.
- Sorts using weighted IMDb score.

---

### ⭐ Popularity-Based Recommendation

- Uses IMDb weighted ratings.
- Provides recommendations for new users.

---

# 📸 Screenshots

Add screenshots here.

```
Home Screen

Movie Recommendations

Popular Movies

Genre Recommendations
```

---

# 📈 Future Improvements

- User Authentication
- Watchlist Support
- TMDB API Integration
- Trailer Recommendations
- Movie Reviews
- Similar Actors Recommendation
- Recommendation Explanation
- Kubernetes Deployment
- CI/CD Pipeline using GitHub Actions
- Redis Caching
- Model Retraining Pipeline

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push

```bash
git push origin feature-name
```

5. Create a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Aryan Choudhary**

- GitHub: https://github.com/Aryannchoudhary
- LinkedIn: https://www.linkedin.com/in/aryan176

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.