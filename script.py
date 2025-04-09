


from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import get_close_matches

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
CONFIG = {
    'default_thumbnail': 'https://placehold.co/300x200?text=No+Image',
    'max_recommendations': 50,
    'max_search_results': 20,
    'max_similar_books': 10,
    'similarity_threshold': 0.3
}

# Load and preprocess data
def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Standard cleaning
    df['title'] = df['title'].fillna('Unknown Title')
    df['authors'] = df['authors'].fillna('Unknown Author')
    df['description'] = df['description'].fillna('')
    df['categories'] = df['categories'].fillna('').astype(str).str.lower()
    df['thumbnail'] = df['thumbnail'].fillna(CONFIG['default_thumbnail'])
    
    # Numeric fields
    numeric_fields = ['average_rating', 'num_pages', 'published_year']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
    
    # Add unique IDs if not present
    if 'id' not in df.columns:
        df['id'] = df.reset_index().index.astype(str)
    
    return df

# Genre configuration
GENRE_SYSTEM = {
    'categories': ['fiction', 'non-fiction', 'science', 'travel'],
    'genres': {
        'fiction': ['fantasy', 'historical fiction', 'mystery fiction', 'adventure'],
        'non-fiction': ['science & nature', 'historical accounts'],
        'science': ['science & technology', 'science education', 'environmental science'],
        'travel': ['travel narratives', 'cultural travel', 'adventure travel']
    },
    'synonyms': {
        'biography & memoirs': ['biography', 'memoir', 'autobiography'],
        'science & nature': ['science', 'nature', 'biology', 'physics'],
        'historical accounts': ['history', 'historical', 'chronicle'],
        'mystery fiction': ['mystery', 'detective', 'crime'],
        'fantasy': ['fantasy', 'magic', 'mythical'],
        'historical fiction': ['historical', 'history'],
        'science & technology': ['science', 'technology', 'engineering'],
        'science education': ['science', 'education', 'learning'],
        'travel narratives': ['travel', 'journey', 'voyage'],
        'cultural travel': ['culture', 'travel', 'heritage']
    }
}

# Text processing utilities
class TextProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def tokenize(self, text):
        return set(re.findall(r'\w+', text.lower()))
    
    def flexible_match(self, text, query, threshold=CONFIG['similarity_threshold']):
        query_terms = GENRE_SYSTEM['synonyms'].get(query.lower(), [query.lower()])
        text_tokens = self.tokenize(text)
        
        for term in query_terms:
            term_tokens = self.tokenize(term)
            if not term_tokens:
                continue
            match_ratio = len(text_tokens.intersection(term_tokens)) / len(term_tokens)
            if match_ratio >= threshold:
                return True
        return False
    
    def calculate_similarity(self, query, documents):
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        query_vector = self.vectorizer.transform([query])
        return cosine_similarity(query_vector, tfidf_matrix).flatten()

text_processor = TextProcessor()

# Load data
try:
    df = load_and_clean_data('cleaned_data.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame(columns=['title', 'authors', 'description', 'categories', 'thumbnail'])

# API Endpoints
@app.route('/categories', methods=['GET'])
def get_categories():
    return jsonify(GENRE_SYSTEM['categories'])

@app.route('/genres', methods=['GET', 'POST'])
def get_genres():
    try:
        if request.method == 'POST':
            data = request.get_json()
            category = data.get('category', '').lower().strip()
        else:
            category = request.args.get('category', '').lower().strip()
        
        if not category:
            return jsonify({'error': 'Category parameter required'}), 400
        
        genres = GENRE_SYSTEM['genres'].get(category, [])
        return jsonify({'genres': genres})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend_books():
    try:
        data = request.get_json()
        category = data.get('category', '').lower().strip()
        genre = data.get('genre', '').lower().strip()
        
        if not category or not genre:
            return jsonify({'error': 'Category and genre required'}), 400

        # Filter books using flexible matching
        filtered = df[df.apply(
            lambda row: (
                text_processor.flexible_match(row['categories'], category) and 
                text_processor.flexible_match(row['categories'], genre)
            ), axis=1
        )]
        
        if filtered.empty:
            return jsonify({'error': 'No books found for this genre'}), 404

        # Calculate similarity scores
        documents = filtered['description'] + ' ' + filtered['categories']
        similarities = text_processor.calculate_similarity(genre, documents)
        ranked_indices = similarities.argsort()[::-1]

        # Prepare results (THIS IS WHERE THE FIX IS)
        results = filtered.iloc[ranked_indices[:CONFIG['max_recommendations']]]  # Added missing ]
        results_clean = results.fillna({
            'average_rating': 0,
            'num_pages': 0,
            'published_year': 0
        }).replace({np.nan: None})
        
        return jsonify({
            'featured': results_clean.head(5).to_dict('records'),
            'recommendations': results_clean.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_books():
    try:
        data = request.get_json()
        query = data.get('query', '').lower().strip()
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400

        # First try exact matches
        exact_matches = df[df['title'].str.lower().str.contains(query, regex=False)]
        
        # If not enough results, try fuzzy matching
        if len(exact_matches) < 5:
            close_matches = get_close_matches(query, df['title'].str.lower().tolist(), n=5, cutoff=0.6)
            fuzzy_matches = df[df['title'].str.lower().isin(close_matches)]
            results = pd.concat([exact_matches, fuzzy_matches]).drop_duplicates()
        else:
            results = exact_matches
        
        # If still not enough, use TF-IDF search
        if len(results) < 5:
            documents = df['title'] + ' ' + df['authors'] + ' ' + df['categories']
            similarities = text_processor.calculate_similarity(query, documents)
            ranked_indices = similarities.argsort()[::-1]
            tfidf_results = df.iloc[ranked_indices[:CONFIG['max_search_results']]]
            results = pd.concat([results, tfidf_results]).drop_duplicates()
        
        if results.empty:
            return jsonify({'error': 'No books found matching your search'}), 404

        # Clean and return results
        results_clean = results.fillna({
            'average_rating': 0,
            'num_pages': 0,
            'published_year': 0
        }).replace({np.nan: None})
        
        return jsonify({
            'results': results_clean.head(CONFIG['max_search_results']).to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)


# .........................................................................FINALCODE.........................
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re
# from difflib import get_close_matches

# app = Flask(__name__)
# CORS(app)

# # Configuration
# CONFIG = {
#     'default_thumbnail': 'https://placehold.co/300x200?text=No+Image',
#     'max_recommendations': 50,
#     'max_search_results': 20,
#     'max_similar_books': 10,
#     'similarity_threshold': 0.3
# }

# # Load and preprocess data
# def load_and_clean_data(filepath):
#     df = pd.read_csv(filepath)
    
#     # Standard cleaning
#     df['title'] = df['title'].fillna('Unknown Title')
#     df['authors'] = df['authors'].fillna('Unknown Author')
#     df['description'] = df['description'].fillna('')
#     df['categories'] = df['categories'].fillna('').astype(str).str.lower()
#     df['thumbnail'] = df['thumbnail'].fillna(CONFIG['default_thumbnail'])
    
#     # Numeric fields
#     numeric_fields = ['average_rating', 'num_pages', 'published_year']
#     for field in numeric_fields:
#         df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
    
#     # Add unique IDs if not present
#     if 'id' not in df.columns:
#         df['id'] = df.reset_index().index.astype(str)
    
#     return df

# df = load_and_clean_data('cleaned_data.csv')

# # Genre configuration
# GENRE_SYSTEM = {
#     'categories': ['fiction', 'non-fiction', 'science', 'travel'],
#     'genres': {
#         'fiction': ['fantasy', 'historical fiction', 'mystery fiction', 'adventure'],
#         'non-fiction': ['science & nature', 'historical accounts'],
#         'science': ['science & technology', 'science education', 'environmental science'],
#         'travel': ['travel narratives', 'cultural travel', 'adventure travel']
#     },
#     'synonyms': {
#         'biography & memoirs': ['biography', 'memoir', 'autobiography'],
#         'science & nature': ['science', 'nature', 'biology', 'physics'],
#         'historical accounts': ['history', 'historical', 'chronicle'],
#         'mystery fiction': ['mystery', 'detective', 'crime'],
#         'fantasy': ['fantasy', 'magic', 'mythical'],
#         'historical fiction': ['historical', 'history'],
#         'science & technology': ['science', 'technology', 'engineering'],
#         'science education': ['science', 'education', 'learning'],
#         'travel narratives': ['travel', 'journey', 'voyage'],
#         'cultural travel': ['culture', 'travel', 'heritage']
#     }
# }

# # Text processing utilities
# class TextProcessor:
#     def __init__(self):
#         self.vectorizer = TfidfVectorizer(stop_words='english')
    
#     def tokenize(self, text):
#         return set(re.findall(r'\w+', text.lower()))
    
#     def flexible_match(self, text, query, threshold=CONFIG['similarity_threshold']):
#         query_terms = GENRE_SYSTEM['synonyms'].get(query.lower(), [query.lower()])
#         text_tokens = self.tokenize(text)
        
#         for term in query_terms:
#             term_tokens = self.tokenize(term)
#             if not term_tokens:
#                 continue
#             match_ratio = len(text_tokens.intersection(term_tokens)) / len(term_tokens)
#             if match_ratio >= threshold:
#                 return True
#         return False
    
#     def calculate_similarity(self, query, documents):
#         tfidf_matrix = self.vectorizer.fit_transform(documents)
#         query_vector = self.vectorizer.transform([query])
#         return cosine_similarity(query_vector, tfidf_matrix).flatten()

# text_processor = TextProcessor()

# # API Endpoints
# @app.route('/categories', methods=['GET'])
# def get_categories():
#     return jsonify(GENRE_SYSTEM['categories'])

# @app.route('/genres/<category>', methods=['GET'])
# def get_genres(category):
#     return jsonify(GENRE_SYSTEM['genres'].get(category.lower(), []))

# @app.route('/recommend', methods=['POST'])
# def recommend_books():
#     try:
#         data = request.get_json()
#         category = data.get('category', '').lower().strip()
#         genre = data.get('genre', '').lower().strip()
        
#         if not category or not genre:
#             return jsonify({'error': 'Category and genre required'}), 400

#         # Filter books using flexible matching
#         filtered = df[df.apply(
#             lambda row: (
#                 text_processor.flexible_match(row['categories'], category) and 
#                 text_processor.flexible_match(row['categories'], genre)
#             ), axis=1
#         )]
        
#         if filtered.empty:
#             return jsonify({'error': 'No books found for this genre'}), 404

#         # Calculate similarity scores
#         documents = filtered['description'] + ' ' + filtered['categories']
#         similarities = text_processor.calculate_similarity(genre, documents)
#         ranked_indices = similarities.argsort()[::-1]

#         # Prepare results
#         results = filtered.iloc[ranked_indices[:CONFIG['max_recommendations']]]
#         results_clean = results.fillna({
#             'average_rating': 0,
#             'num_pages': 0,
#             'published_year': 0
#         }).replace({np.nan: None})
        
#         return jsonify({
#             'top_5': results_clean.head(5).to_dict('records'),
#             'top_50': results_clean.to_dict('records')
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/search', methods=['POST'])
# def search_books():
#     try:
#         data = request.get_json()
#         query = data.get('query', '').lower().strip()
        
#         if not query:
#             return jsonify({'error': 'Search query required'}), 400

#         # First try exact matches
#         exact_matches = df[df['title'].str.lower().str.contains(query, regex=False)]
        
#         # If not enough results, try fuzzy matching
#         if len(exact_matches) < 5:
#             close_matches = get_close_matches(query, df['title'].str.lower().tolist(), n=5, cutoff=0.6)
#             fuzzy_matches = df[df['title'].str.lower().isin(close_matches)]
#             results = pd.concat([exact_matches, fuzzy_matches]).drop_duplicates()
#         else:
#             results = exact_matches
        
#         # If still not enough, use TF-IDF search
#         if len(results) < 5:
#             documents = df['title'] + ' ' + df['authors'] + ' ' + df['categories']
#             similarities = text_processor.calculate_similarity(query, documents)
#             ranked_indices = similarities.argsort()[::-1]
#             tfidf_results = df.iloc[ranked_indices[:CONFIG['max_search_results']]]
#             results = pd.concat([results, tfidf_results]).drop_duplicates()
        
#         if results.empty:
#             return jsonify({'error': 'No books found matching your search'}), 404

#         # Clean and return results
#         results_clean = results.fillna({
#             'average_rating': 0,
#             'num_pages': 0,
#             'published_year': 0
#         }).replace({np.nan: None})
        
#         return jsonify({
#             'results': results_clean.head(CONFIG['max_search_results']).to_dict('records')
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/similar', methods=['POST'])
# def get_similar_books():
#     try:
#         data = request.get_json()
#         book_id = data.get('book_id')
        
#         if not book_id:
#             return jsonify({'error': 'Book ID required'}), 400

#         # Find the book
#         book = df[df['id'] == book_id].iloc[0]
        
#         # Calculate similarity
#         features = df['description'] + ' ' + df['categories']
#         similarities = text_processor.calculate_similarity(
#             book['description'] + ' ' + book['categories'],
#             features
#         )
        
#         # Get most similar books (excluding the book itself)
#         similar_indices = similarities.argsort()[::-1][1:CONFIG['max_similar_books']+1]
#         similar_books = df.iloc[similar_indices].fillna({
#             'average_rating': 0,
#             'num_pages': 0
#         }).replace({np.nan: None})
        
#         return jsonify({
#             'similar_books': similar_books.to_dict('records')
#         })
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)































































# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import re

# app = Flask(__name__)
# CORS(app)

# # Load and preprocess data
# df = pd.read_csv("cleaned_data.csv")

# # Data cleaning and preparation
# def clean_data(df):
#     # Fill missing values
#     df["title"] = df["title"].fillna("Unknown Title")
#     df["authors"] = df["authors"].fillna("Unknown Author")
#     df["description"] = df["description"].fillna("")
#     df["categories"] = df["categories"].fillna("").astype(str).str.lower()
#     df["average_rating"] = pd.to_numeric(df["average_rating"], errors='coerce').fillna(0)
#     df["num_pages"] = pd.to_numeric(df["num_pages"], errors='coerce').fillna(0)
#     df["published_year"] = pd.to_numeric(df["published_year"], errors='coerce').fillna(0)
#     df["thumbnail"] = df["thumbnail"].fillna("https://placehold.co/300x200?text=No+Image")
    
#     # Add unique IDs if not present
#     if 'id' not in df.columns:
#         df['id'] = df.reset_index().index.astype(str)
    
#     return df

# df = clean_data(df)

# # Genre configuration
# PREDEFINED_CATEGORIES = ["fiction", "non-fiction", "science", "travel"]
# PREDEFINED_GENRES = {
#     "fiction": ["fantasy", "historical fiction", "mystery fiction", "adventure"],
#     "non-fiction": ["science & nature", "historical accounts"],
#     "science": ["science & technology", "science education", "environmental science"],
#     "travel": ["travel narratives", "cultural travel", "adventure travel"]
# }

# # Text processing and similarity
# vectorizer = TfidfVectorizer(stop_words="english")

# def get_similarity_scores(query, documents):
#     """Calculate cosine similarity between query and documents"""
#     tfidf_matrix = vectorizer.fit_transform(documents)
#     query_vector = vectorizer.transform([query])
#     return cosine_similarity(query_vector, tfidf_matrix).flatten()

# def flexible_match(row, category, genre, threshold=0.3):
#     """Flexible matching for categories and genres"""
#     categories = str(row["categories"]).lower()
#     return (category in categories and genre in categories)

# # API Endpoints
# @app.route('/categories', methods=['GET'])
# def get_categories():
#     return jsonify({"categories": PREDEFINED_CATEGORIES})

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No data provided"}), 400
            
#         category = data.get("category", "").lower().strip()
#         genre = data.get("genre", "").lower().strip()
        
#         if not category or not genre:
#             return jsonify({"error": "Category and genre required"}), 400

#         # Filter books using flexible matching
#         filtered = df[df.apply(
#             lambda row: flexible_match(row, category, genre), 
#             axis=1
#         )]
        
#         if filtered.empty:
#             return jsonify({"error": "No books found. Try another genre."}), 404

#         # Calculate similarity scores
#         documents = filtered["description"] + " " + filtered["categories"]
#         similarities = get_similarity_scores(genre, documents)
#         ranked_indices = similarities.argsort()[::-1]

#         # Prepare results with proper null handling
#         result_data = filtered.iloc[ranked_indices[:50]].fillna({
#             'average_rating': 0,
#             'num_pages': 0,
#             'published_year': 0
#         }).replace({np.nan: None})
        
#         return jsonify({
#             "top_5": result_data.head(5).to_dict(orient="records"),
#             "top_50": result_data.to_dict(orient="records")
#         })
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/search', methods=['POST'])
# def search_books():
#     try:
#         data = request.get_json()
#         query = data.get('query', '').lower().strip()
        
#         if not query:
#             return jsonify({"error": "Search query required"}), 400

#         # Search across multiple fields using TF-IDF
#         documents = df['title'] + " " + df['authors'] + " " + df['categories']
#         similarities = get_similarity_scores(query, documents)
#         ranked_indices = similarities.argsort()[::-1]
        
#         # Get top 20 matches and clean data
#         results = df.iloc[ranked_indices[:20]].fillna({
#             'average_rating': 0,
#             'num_pages': 0,
#             'published_year': 0
#         }).replace({np.nan: None})
        
#         return jsonify({
#             "results": results.to_dict(orient="records")
#         })
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/similar', methods=['POST'])
# def similar_books():
#     try:
#         data = request.get_json()
#         book_id = data.get('book_id')
        
#         if not book_id:
#             return jsonify({"error": "Book ID required"}), 400

#         # Find the selected book
#         book = df[df['id'] == book_id].iloc[0]
        
#         # Calculate similarity based on combined features
#         features = df['description'] + " " + df['categories']
#         similarities = get_similarity_scores(
#             book['description'] + " " + book['categories'],
#             features
#         )
        
#         # Get most similar books (excluding the book itself)
#         similar_indices = similarities.argsort()[::-1][1:11]
#         similar_books = df.iloc[similar_indices].fillna({
#             'average_rating': 0,
#             'num_pages': 0
#         }).replace({np.nan: None})
        
#         return jsonify({
#             "similar_books": similar_books.to_dict(orient="records")
#         })
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)
# ..........................................................both above for search based  and below for browsing based..................
# import re
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from difflib import get_close_matches

# app = Flask(__name__)
# CORS(app)

# # Load and preprocess data
# df = pd.read_csv("cleaned_data.csv")
# df["categories"] = df["categories"].astype(str).str.lower()
# df["description"] = df["description"].fillna("")

# # Handle NaN values in critical columns
# df["title"] = df["title"].fillna("Unknown Title")
# df["authors"] = df["authors"].fillna("Unknown Author")
# df["average_rating"] = df["average_rating"].fillna(0)
# df["num_pages"] = df["num_pages"].fillna(0)
# df["published_year"] = df["published_year"].fillna(0)
# df["thumbnail"] = df["thumbnail"].fillna("https://placehold.co/300x200?text=No+Image")

# # Add unique IDs if not present
# if 'id' not in df.columns:
#     df['id'] = df.reset_index().index.astype(str)

# # Genre configuration
# PREDEFINED_CATEGORIES = ["fiction", "non-fiction", "science", "travel"]
# PREDEFINED_GENRES = {
#     "fiction": [
#         "fantasy",
#         "historical fiction", 
#         "mystery fiction",
#         "adventure"
#     ],
#     "non-fiction": [
#         "science & nature",
#         "historical accounts"
#     ],
#     "science": [
#         "science & technology",
#         "science education",
#         "environmental science"
#     ],
#     "travel": [
#         "travel narratives",
#         "cultural travel",
#         "adventure travel"
#     ]
# }

# # Text processing
# def tokenize(text):
#     return set(re.findall(r'\w+', text.lower()))

# def flexible_match(field, query, threshold=0.3):
#     synonym_map = {
#         # Non-fiction genres
#         "biography & memoirs": ["biography", "memoir", "autobiography"],
#         "science & nature": ["science", "nature", "biology", "physics"],
#         "historical accounts": ["history", "historical", "chronicle"],
        
#         # Fiction genres
#         "mystery fiction": ["mystery", "detective", "crime"],
#         "fantasy": ["fantasy", "magic", "mythical"],
#         "historical fiction": ["historical", "history"],
        
#         # Science genres
#         "science & technology": ["science", "technology", "engineering"],
#         "science education": ["science", "education", "learning"],
        
#         # Travel genres
#         "travel narratives": ["travel", "journey", "voyage"],
#         "cultural travel": ["culture", "travel", "heritage"]
#     }
#     query_terms = synonym_map.get(query.lower(), [query.lower()])
#     field_tokens = tokenize(field)
    
#     for term in query_terms:
#         term_tokens = tokenize(term)
#         if not term_tokens:
#             continue
#         match_ratio = len(field_tokens.intersection(term_tokens)) / len(term_tokens)
#         if match_ratio >= threshold:
#             return True
#     return False

# # API Endpoints
# @app.route('/categories', methods=['GET'])
# def get_categories():
#     return jsonify({"categories": PREDEFINED_CATEGORIES})

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No data provided"}), 400
            
#         category = data.get("category", "").lower().strip()
#         genre = data.get("genre", "").lower().strip()
        
#         if not category or not genre:
#             return jsonify({"error": "Category and genre required"}), 400

#         # Filter books
#         filtered = df[df.apply(
#             lambda row: (
#                 flexible_match(row["categories"], category) and 
#                 flexible_match(row["categories"], genre)
#             ), axis=1
#         )]
        
#         if filtered.empty:
#             return jsonify({"error": "No books found. Try another genre."}), 404

#         # TF-IDF similarity
#         vectorizer = TfidfVectorizer(stop_words="english")
#         tfidf_matrix = vectorizer.fit_transform(filtered["description"])
#         query_vector = vectorizer.transform([genre])
#         similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#         ranked_indices = similarities.argsort()[::-1]

#         # Prepare results with null checks
#         result_data = filtered.iloc[ranked_indices[:50]].replace({pd.NA: None})
#         top_5 = result_data.head(5).to_dict(orient="records")
#         top_50 = result_data.to_dict(orient="records")

#         return jsonify({
#             "top_5": top_5,
#             "top_50": top_50
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/search', methods=['POST'])
# def search_books():
#     try:
#         data = request.get_json()
#         query = data.get('query', '').lower().strip()
        
#         if not query:
#             return jsonify({"error": "Search query required"}), 400

#         # First try exact matches
#         exact_matches = df[df['title'].str.lower().str.contains(query, regex=False)]
        
#         # If not enough results, try fuzzy matching
#         if len(exact_matches) < 5:
#             all_titles = df['title'].str.lower().tolist()
#             close_matches = get_close_matches(query, all_titles, n=5, cutoff=0.6)
#             fuzzy_matches = df[df['title'].str.lower().isin(close_matches)]
#             results = pd.concat([exact_matches, fuzzy_matches]).drop_duplicates()
#         else:
#             results = exact_matches
        
#         if results.empty:
#             return jsonify({"error": "No books found matching your search"}), 404

#         return jsonify({
#             "results": results.head(10).replace({pd.NA: None}).to_dict(orient="records")
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/similar', methods=['POST'])
# def similar_books():
#     try:
#         data = request.get_json()
#         book_id = data.get('book_id')
        
#         if not book_id:
#             return jsonify({"error": "Book ID required"}), 400

#         # Find the selected book
#         book = df[df['id'] == book_id].iloc[0]
        
#         # Calculate similarity based on description and categories
#         vectorizer = TfidfVectorizer(stop_words="english")
#         features = df['description'] + " " + df['categories']
#         tfidf_matrix = vectorizer.fit_transform(features)
#         book_features = book['description'] + " " + book['categories']
#         book_vector = vectorizer.transform([book_features])
        
#         similarities = cosine_similarity(book_vector, tfidf_matrix).flatten()
#         similar_indices = similarities.argsort()[::-1][1:11]  # exclude the book itself
        
#         similar_books = df.iloc[similar_indices]
        
#         return jsonify({
#             "similar_books": similar_books.replace({pd.NA: None}).to_dict(orient="records")
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)


# ///////////////////////////////////////////////////////////////////////////////////////////////
# import re
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)
# CORS(app)

# # Load and preprocess data
# df = pd.read_csv("cleaned_data.csv")
# df["categories"] = df["categories"].astype(str).str.lower()
# df["description"] = df["description"].fillna("")

# # Handle NaN values in critical columns
# df["title"] = df["title"].fillna("Unknown Title")
# df["authors"] = df["authors"].fillna("Unknown Author")
# df["average_rating"] = df["average_rating"].fillna(0)
# df["num_pages"] = df["num_pages"].fillna(0)
# df["published_year"] = df["published_year"].fillna(0)
# df["thumbnail"] = df["thumbnail"].fillna("https://via.placeholder.com/150x200?text=No+Image")

# # Genre configuration
# PREDEFINED_CATEGORIES = ["fiction", "non-fiction", "science", "travel"]
# PREDEFINED_GENRES = {
#     "fiction": [
#         "fantasy",
#         "historical fiction", 
#         "Mystery fiction",
#         "adventure"
#     ],
#     "non-fiction": [
#         "science & nature",
#         "historical accounts"
#         # "self-help",
#         # "religion & spirituality",
#         # "travel & culture"
#     ],
#     "science": [
#         "science & technology",
#         "science education",
#         "environmental science"
#     ],
#     "travel": [
#         "travel narratives",
#         "cultural travel",
#         "adventure travel"
#     ]
# }

# # Text processing
# def tokenize(text):
#     return set(re.findall(r'\w+', text.lower()))

# def flexible_match(field, query, threshold=0.3):
#     synonym_map = {
#         "mystery & detective": ["mystery", "detective", "crime"],
#         "fantasy": ["fantasy", "arthurian", "magic"],
#         "historical fiction": ["historical", "history"],
#         "biography & memoirs": ["biography", "autobiography", "memoir"],
#         "self-help": ["self help", "psychology", "emotional"],
#         "science & technology": ["science", "technology", "research"],
#         "travel & culture": ["travel", "culture", "geography"]
#     }
#     query_terms = synonym_map.get(query.lower(), [query.lower()])
#     field_tokens = tokenize(field)
    
#     for term in query_terms:
#         term_tokens = tokenize(term)
#         if not term_tokens:
#             continue
#         match_ratio = len(field_tokens.intersection(term_tokens)) / len(term_tokens)
#         if match_ratio >= threshold:
#             return True
#     return False

# # API Endpoints
# @app.route('/categories', methods=['GET'])
# def get_categories():
#     return jsonify({"categories": PREDEFINED_CATEGORIES})

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     try:
#         data = request.get_json()
#         if not data:
#             return jsonify({"error": "No data provided"}), 400
            
#         category = data.get("category", "").lower().strip()
#         genre = data.get("genre", "").lower().strip()
        
#         if not category or not genre:
#             return jsonify({"error": "Category and genre required"}), 400

#         # Filter books
#         filtered = df[df.apply(
#             lambda row: (
#                 flexible_match(row["categories"], category) and 
#                 flexible_match(row["categories"], genre)
#             ), axis=1
#         )]
        
#         if filtered.empty:
#             return jsonify({"error": "No books found. Try another genre."}), 404

#         # TF-IDF similarity
#         vectorizer = TfidfVectorizer(stop_words="english")
#         tfidf_matrix = vectorizer.fit_transform(filtered["description"])
#         query_vector = vectorizer.transform([genre])
#         similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#         ranked_indices = similarities.argsort()[::-1]

#         # Prepare results with null checks
#         top_5 = filtered.iloc[ranked_indices[:5]].replace({pd.NA: None}).to_dict(orient="records")
#         top_50 = filtered.iloc[ranked_indices[:50]].replace({pd.NA: None}).to_dict(orient="records")

#         return jsonify({
#             "top_5": top_5,
#             "top_50": top_50
#         })
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)


# import re
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)
# CORS(app)

# # Load dataset
# df = pd.read_csv("cleaned_books.csv")
# df["categories"] = df["categories"].astype(str).str.lower()
# df["description"] = df["description"].fillna("")

# def tokenize(text):
#     return set(re.findall(r'\w+', text.lower()))

# def flexible_match(field, query, threshold=0.2):
#     field_tokens = tokenize(field)
#     query_tokens = tokenize(query)
#     if not query_tokens:
#         return False
#     match_ratio = len(field_tokens.intersection(query_tokens)) / len(query_tokens)
#     return match_ratio >= threshold

# predefined_categories = ["fiction", "non-fiction", "science", "travel"]
# predefined_genres =  {
#     "fiction": ["thriller fiction", "adventure", "historical fiction", "fantasy fiction"],
#     "non-fiction": ["self-help", "biographies", "historical accounts"],
#     "science": ["science education", "research", "science fiction"],
#     "travel": ["travel narrative", "autobiography", "cultural travel"]
# }


# @app.route('/categories', methods=['GET'])
# def get_categories():
#     return jsonify({"categories": predefined_categories})

# @app.route('/genres', methods=['POST'])
# def get_genres():
#     data = request.get_json()
#     category = data.get("category", "").lower()
#     return jsonify({"genres": predefined_genres.get(category, [])})

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.get_json()
#     category = data.get("category", "").lower()
#     genre = data.get("genre", "").lower()
#     search_text = data.get("search_text", "").strip()

#     if not category or not genre:
#         return jsonify({"error": "Category and Genre are required"}), 400

#     # Stricter genre matching
#     filtered = df[df.apply(
#         lambda row: flexible_match(row["categories"], category, threshold=0.2)
#                  and flexible_match(row["categories"], genre, threshold=0.3),
#         axis=1
#     )]

#     if filtered.empty:
#         return jsonify({"error": "No matching books found."})

#     reference_text = search_text if search_text else genre

#     # TF-IDF calculation
#     vectorizer = TfidfVectorizer(stop_words="english")
#     tfidf_matrix = vectorizer.fit_transform(filtered["description"])
#     query_vector = vectorizer.transform([reference_text])
#     similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#     ranked_indices = similarities.argsort()[::-1]

#     # Split results
#     top_5_indices = ranked_indices[:5]
#     top_50_indices = ranked_indices[:50]

#     return jsonify({
#         "top_5_genre": filtered.iloc[top_5_indices].fillna("").to_dict(orient="records"),
#         "top_50": filtered.iloc[top_50_indices].fillna("").to_dict(orient="records")
#     })

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)


# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# import re
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)
# CORS(app)  # Allow all origins

# # Load dataset
# df = pd.read_csv("cleaned_books.csv")
# print("Columns in CSV:", df.columns.tolist())

# # Ensure consistency: convert "categories" to lowercase strings and fill missing descriptions.
# df["categories"] = df["categories"].astype(str).str.lower()
# df["description"] = df["description"].fillna("")

# # Helper: Tokenize text into a set of word tokens.
# def tokenize(text):
#     return set(re.findall(r'\w+', text.lower()))

# # Flexible matching function:
# def flexible_match(field, query, threshold=0.2):
#     field_tokens = tokenize(field)
#     query_tokens = tokenize(query)
#     if not query_tokens:
#         return False
#     match_ratio = len(field_tokens.intersection(query_tokens)) / len(query_tokens)
#     return match_ratio >= threshold

# # Predefined top-level categories and fixed genres (4 each)
# predefined_categories = ["fiction", "non-fiction", "science", "travel"]
# predefined_genres = {
#     "fiction": ["thriller fiction","adventure", "historical fiction", "fantasy fiction"],
#     "non-fiction": ["self-help", "biographies", "historical accounts"],
#     "science": ["science education", "research", "science fiction"],
#     "travel": ["travel narrative", "autobiography", "cultural travel"]
# }

# @app.route('/categories', methods=['GET'])
# def get_categories():
#     return jsonify({"categories": predefined_categories})

# @app.route('/genres', methods=['POST'])
# def get_genres():
#     data = request.get_json()
#     category = data.get("category", "").lower()
#     if not category or category not in predefined_genres:
#         return jsonify({"error": "Invalid category provided"}), 400
#     return jsonify({"genres": predefined_genres[category]})

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.get_json()
#     category = data.get("category", "").lower()
#     genre = data.get("genre", "").lower()
#     search_text = data.get("search_text", "").strip()
    
#     if not category or not genre:
#         return jsonify({"error": "Category and Genre are required"}), 400

#     # Use flexible matching for both category and genre.
#     filtered = df[df.apply(
#         lambda row: flexible_match(row["categories"], category, threshold=0.2)
#                  and flexible_match(row["categories"], genre, threshold=0.2),
#         axis=1
#     )]

#     print("Filtering with category:", category, "and genre:", genre)
#     print("Filtered count:", len(filtered))
#     if filtered.empty:
#         return jsonify({"error": "No matching books found for the selected category and genre."})
    
#     reference_text = search_text if search_text else filtered.iloc[0]["description"]

#     # TF-IDF vectorization on descriptions.
#     vectorizer = TfidfVectorizer(stop_words="english")
#     descriptions = filtered["description"].tolist()
#     tfidf_matrix = vectorizer.fit_transform(descriptions)
#     query_vector = vectorizer.transform([reference_text])
#     similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#     ranked_indices = similarities.argsort()[::-1]

#     if not search_text:
#         ranked_indices = ranked_indices[ranked_indices != 0]

#     top_n = 50
#     recommended_books = filtered.iloc[ranked_indices].head(top_n)
#     recommended_books = recommended_books.fillna("")
#     recommendations = recommended_books[[
#         "title", "subtitle", "authors", "categories", "description",
#         "published_year", "average_rating", "num_pages", "ratings_count"
#     ]].to_dict(orient="records")

#     return jsonify({"recommendations": recommendations})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)






# import re
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# app = Flask(__name__)
# CORS(app)  # Allow all origins

# # Load your dataset once (ensure the CSV file is in the same folder)
# df = pd.read_csv("cleaned_books.csv")
# print("Columns in CSV:", df.columns.tolist())

# # Convert the "categories" column to string and lowercase for consistency.
# df["categories"] = df["categories"].astype(str).str.lower()
# df["description"] = df["description"].fillna("")

# # Tokenize helper function: returns a set of word tokens
# def tokenize(text):
#     return set(re.findall(r'\w+', text.lower()))

# @app.route('/categories', methods=['GET'])
# def get_categories():
#     top_categories = ["fiction", "non-fiction", "science"]
#     return jsonify({"categories": top_categories})

# @app.route('/genres', methods=['POST'])
# def get_genres():
#     data = request.get_json()
#     category = data.get("category", "").lower()
#     if not category:
#         return jsonify({"error": "Category not provided"}), 400

#     filtered = df[df["categories"].str.contains(category)]
    
#     genres_set = set()
#     for cat_str in filtered["categories"]:
#         for part in cat_str.split(","):
#             part = part.strip()
#             if part:
#                 genres_set.add(part)
#     genres = list(genres_set)
#     return jsonify({"genres": genres})

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.get_json()
#     category = data.get("category", "").lower()
#     genre = data.get("genre", "").lower()
#     search_text = data.get("search_text", "").strip()

#     if not category or not genre:
#         return jsonify({"error": "Category and Genre are required"}), 400

#     # Improved flexible filter using tokenization.
#     def flexible_filter(x, selected_genre, threshold=0.3):
#         x_tokens = tokenize(x)
#         genre_tokens = tokenize(selected_genre)
#         if not genre_tokens:
#             return False
#         match_ratio = len(genre_tokens.intersection(x_tokens)) / len(genre_tokens)
#         return match_ratio >= threshold

#     # Apply filtering: book's categories must contain the top-level category
#     # and pass the flexible filter for the selected genre.
#     filtered = df[
#         df["categories"].str.contains(category) & 
#         df["categories"].apply(lambda x: flexible_filter(x, genre, threshold=0.3))
#     ]

#     print("Filtering with category:", category, "and genre:", genre)
#     print("Filtered count:", len(filtered))
    
#     if filtered.empty:
#         return jsonify({"error": "No matching books found for the selected category and genre."})

#     if not search_text:
#         reference_text = filtered.iloc[0]["description"]
#     else:
#         reference_text = search_text

#     vectorizer = TfidfVectorizer(stop_words="english")
#     descriptions = filtered["description"].tolist()
#     tfidf_matrix = vectorizer.fit_transform(descriptions)
#     query_vector = vectorizer.transform([reference_text])
#     similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
#     ranked_indices = similarities.argsort()[::-1]

#     if not search_text:
#         ranked_indices = ranked_indices[ranked_indices != 0]

#     top_n = 5
#     recommended_books = filtered.iloc[ranked_indices].head(top_n)
    
#     # Fill NaN values so that JSON can be produced properly
#     recommended_books = recommended_books.fillna("")
#     recommendations = recommended_books[[
#         "title", "subtitle", "authors", "categories", "description",
#         "published_year", "average_rating", "num_pages", "ratings_count"
#     ]].to_dict(orient="records")

#     return jsonify({"recommendations": recommendations})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)





# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pandas as pd

# app = Flask(__name__)
# CORS(app)  # Allow all origins

# # Load the dataset once at startup
# df = pd.read_csv("cleaned_books.csv")
# # The CSV has a column "categories". We'll work with that.
# # Convert the column to string and lowercase for consistency.
# df["categories"] = df["categories"].astype(str).str.lower()

# # Define your own top-level categories.
# # Adjust these keywords as needed to match the contents of your "categories" column.
# top_categories = ["fiction", "non-fiction", "science"]

# @app.route('/categories', methods=['GET'])
# def get_categories():
#     # Simply return our predefined top-level categories.
#     return jsonify({"categories": top_categories})

# @app.route('/genres', methods=['POST'])
# def get_genres():
#     data = request.get_json()
#     category = data.get("category", "").lower()
#     if not category:
#         return jsonify({"error": "Category not provided"}), 400

#     # Filter rows where the "categories" field contains the top-level category keyword.
#     filtered = df[df["categories"].str.contains(category)]
    
#     # Now extract sub-genres.
#     # We assume that the "categories" field may contain comma-separated values.
#     genres_set = set()
#     for cat_str in filtered["categories"]:
#         # Split by comma and strip whitespace
#         for part in cat_str.split(","):
#             part = part.strip()
#             if part:
#                 genres_set.add(part)
#     genres = list(genres_set)
#     return jsonify({"genres": genres})

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.get_json()
#     category = data.get("category", "").lower()
#     genre = data.get("genre", "").lower()
#     if not category or not genre:
#         return jsonify({"error": "Category and Genre are required"}), 400

#     # Filter rows whose "categories" field contains BOTH the top-level category keyword and the selected genre.
#     filtered = df[
#         df["categories"].str.contains(category) & 
#         df["categories"].str.contains(genre)
#     ]
#     # For simplicity, return the first 5 book titles as recommendations.
#     recommendations = filtered["title"].head(5).tolist()
#     return jsonify({"recommendations": recommendations})

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)




# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import json

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# # 🔹 Load dataset (make sure 'cleaned_books.csv' exists in the same directory)
# df = pd.read_csv("cleaned_books.csv")

# # 🔹 Convert title to lowercase for matching
# df["title"] = df["title"].str.lower()

# # 🔹 Fill missing descriptions
# df["description"] = df["description"].fillna("")

# # 🔹 TF-IDF Vectorization
# tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
# tfidf_matrix = tfidf.fit_transform(df["description"])

# # 🔹 Compute Cosine Similarity
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# def clean_field(value):
#     """Convert NaN values to empty string."""
#     if pd.isna(value):
#         return ""
#     return value

# # ✅ Improved Function to Get Book Recommendations with Partial Matching
# def get_book_recommendations(title):
#     title = title.lower().strip()  # Normalize input
    
#     # Use partial matching to find books whose title contains the input string
#     matches = df[df["title"].str.contains(title, na=False)]
    
#     if matches.empty:
#         print(f"⚠️ No book titles found containing '{title}'!")
#         return []
    
#     # Use the first match's index for computing similarity
#     idx = matches.index[0]
    
#     # Get similarity scores & sort them (excluding the input book itself)
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations

#     recommendations = []
#     for i in sim_scores:
#         book = df.iloc[i[0]]
#         recommendations.append({
#             "title": clean_field(book["title"]),
#             "subtitle": clean_field(book.get("subtitle", "")),
#             "authors": clean_field(book.get("authors", "")),
#             "categories": clean_field(book.get("categories", "")),
#             "thumbnail": clean_field(book.get("thumbnail", "")),
#             "description": clean_field(book.get("description", "")),
#             "published_year": clean_field(book.get("published_year", "")),
#             "average_rating": clean_field(book.get("average_rating", "")),
#             "num_pages": clean_field(book.get("num_pages", "")),
#             "ratings_count": clean_field(book.get("ratings_count", "")),
#         })

#     print(f"✅ Found {len(recommendations)} recommendations for '{title}'")
#     return recommendations

# # 🔹 Flask Route for Recommendations
# @app.route("/recommend", methods=["POST"])
# def recommend():
#     data = request.get_json()
#     if not data or "title" not in data:
#         return jsonify({"error": "Missing 'title' in request"}), 400

#     recommended_books = get_book_recommendations(data["title"])
    
#     # Construct the full response data
#     response_data = {
#         "book_title": data["title"],
#         "recommendations": recommended_books
#     }
    
#     # Convert to JSON string for debugging (and ensure valid JSON)
#     response_json = json.dumps(response_data, default=str)
#     print("Final Response JSON:", response_json)

#     return response_json, 200, {"Content-Type": "application/json"}

# if __name__ == "__main__":
#     app.run(debug=True)


# import pandas as pd
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import json

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS

# # 🔹 Load dataset (make sure 'cleaned_books.csv' exists)
# df = pd.read_csv("cleaned_books.csv")  

# # 🔹 Convert title to lowercase for matching
# df["title"] = df["title"].str.lower()

# # 🔹 Fill missing descriptions
# df["description"] = df["description"].fillna("")

# # 🔹 TF-IDF Vectorization
# tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
# tfidf_matrix = tfidf.fit_transform(df["description"])

# # 🔹 Compute Cosine Similarity
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# def clean_field(value):
#     """Convert NaN values to empty string."""
#     if pd.isna(value):
#         return ""
#     return value

# # ✅ Improved Function to Get Book Recommendations
# def get_book_recommendations(title):
#     title = title.lower().strip()  # Normalize input
    
#     # 🔹 Check if title exists
#     if title not in df["title"].values:
#         print(f"⚠️ '{title}' not found in dataset!")
#         return []
    
#     idx = df[df["title"] == title].index[0]
    
#     # 🔹 Get similarity scores & sort them (excluding the input book itself)
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]  # Top 5 recommendations

#     recommendations = []
#     for i in sim_scores:
#         book = df.iloc[i[0]]
#         recommendations.append({
#             "title": clean_field(book["title"]),
#             "subtitle": clean_field(book.get("subtitle", "")),
#             "authors": clean_field(book.get("authors", "")),
#             "categories": clean_field(book.get("categories", "")),
#             "thumbnail": clean_field(book.get("thumbnail", "")),
#             "description": clean_field(book.get("description", "")),
#             "published_year": clean_field(book.get("published_year", "")),
#             "average_rating": clean_field(book.get("average_rating", "")),
#             "num_pages": clean_field(book.get("num_pages", "")),
#             "ratings_count": clean_field(book.get("ratings_count", "")),
#         })

#     print(f"✅ Found {len(recommendations)} recommendations for '{title}'")
#     return recommendations

# # 🔹 Flask Route for Recommendations
# @app.route("/recommend", methods=["POST"])
# def recommend():
#     data = request.get_json()
#     if not data or "title" not in data:
#         return jsonify({"error": "Missing 'title' in request"}), 400

#     recommended_books = get_book_recommendations(data["title"])
    
#     # Construct full response data
#     response_data = {
#         "book_title": data["title"],
#         "recommendations": recommended_books
#     }
    
#     # Convert to JSON string for debugging (and ensure valid JSON)
#     response_json = json.dumps(response_data, default=str)
#     print("Final Response JSON:", response_json)

#     return response_json, 200, {"Content-Type": "application/json"}

# if __name__ == "__main__":
#     app.run(debug=True)






# ns(book_title,top_n=5):
#     book_title = book_title.lower()
#     indices = df[df["title"].str.lower() == book_title].index

#     if len(indices) == 0:
#         return {"error": "Book not found. Please check the title."}

#     book_index = indices[0]  # Get first match
#     similarity_scores = list(enumerate(cosine_sim[book_index]))
#     similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
#     recommended_books = [df.iloc[i[0]]["title"] for i in similarity_scores]

#     return {"book_title": book_title, "recommendations": recommended_books}

# # Flask route for book recommendations
# @app.route('/recommend', methods=['POST'])
# def recommend():
#     data = request.json
#     book_title = data.get("title")

#     if not book_title:
#         return jsonify({"error": "Please provide a book title."}), 400

#     recommendations = get_recommendations(book_title)
#     return jsonify(recommendations)

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from flask_cors import CORS

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app) 
# # Load preprocessed book data
# df = pd.read_csv(r"C:\Users\govin\Downloads\cleaned_data.csv.zip")

# # Ensure missing descriptions are filled
# df["description"] = df["description"].fillna("")

# # Apply TF-IDF Vectorization
# tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
# tfidf_matrix = tfidf.fit_transform(df["description"])

# # Compute cosine similarity
# cosine_sim = cosine_similarity(tfidf_matrix)

# # Recommendation function
# def get_recommendatio