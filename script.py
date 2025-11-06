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

CONFIG = {
    'default_thumbnail': 'https://placehold.co/300x200?text=No+Image',
    'max_recommendations': 50,
    'max_search_results': 20,
    'max_similar_books': 10,
    'similarity_threshold': 0.3
}

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)

    df['title'] = df['title'].fillna('Unknown Title')
    df['authors'] = df['authors'].fillna('Unknown Author')
    df['description'] = df['description'].fillna('')
    df['categories'] = df['categories'].fillna('').astype(str).str.lower()
    df['thumbnail'] = df['thumbnail'].fillna(CONFIG['default_thumbnail'])
    
 
    numeric_fields = ['average_rating', 'num_pages', 'published_year']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)

    if 'id' not in df.columns:
        df['id'] = df.reset_index().index.astype(str)
    
    return df

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

try:
    df = load_and_clean_data('cleaned_data.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame(columns=['title', 'authors', 'description', 'categories', 'thumbnail'])

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

        filtered = df[df.apply(
            lambda row: (
                text_processor.flexible_match(row['categories'], category) and 
                text_processor.flexible_match(row['categories'], genre)
            ), axis=1
        )]
        
        if filtered.empty:
            return jsonify({'error': 'No books found for this genre'}), 404

        documents = filtered['description'] + ' ' + filtered['categories']
        similarities = text_processor.calculate_similarity(genre, documents)
        ranked_indices = similarities.argsort()[::-1]

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

        exact_matches = df[df['title'].str.lower().str.contains(query, regex=False)]
        if len(exact_matches) < 5:
            close_matches = get_close_matches(query, df['title'].str.lower().tolist(), n=5, cutoff=0.6)
            fuzzy_matches = df[df['title'].str.lower().isin(close_matches)]
            results = pd.concat([exact_matches, fuzzy_matches]).drop_duplicates()
        else:
            results = exact_matches
        if len(results) < 5:
            documents = df['title'] + ' ' + df['authors'] + ' ' + df['categories']
            similarities = text_processor.calculate_similarity(query, documents)
            ranked_indices = similarities.argsort()[::-1]
            tfidf_results = df.iloc[ranked_indices[:CONFIG['max_search_results']]]
            results = pd.concat([results, tfidf_results]).drop_duplicates()
        
        if results.empty:
            return jsonify({'error': 'No books found matching your search'}), 404

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