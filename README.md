# 🎬 Movie Recommender System

-A content-based movie recommender system built in Python using pandas, scikit-learn, and NLTK.

-This project recommends similar movies based on genres, keywords, cast, and crew information extracted from the TMDb dataset.

## Features
- Recommend Top 5 Similar Movies
- Text preprocessing and stemming  
- Cosine similarity for ranking recommendations  
- Simple command-line interface

## Dataset
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

These files are included in the repository.

## Installation
**Clone the repository**

git clone https://github.com/gopalsingh18/Movie-Recommender-System.git
cd Movie-Recommender-System

**Install dependencies**

install required packages: 
 pip install -r requirements.txt

Run the recommender script: 
 python movie_recommender.py

You will be prompted to enter a movie title: 
Enter a movie title (or 'exit' to quit):

## Future Improvements

-Build a Streamlit web interface

-Add collaborative filtering models

-Use more advanced NLP techniques for better similarity scoring
