# Restaurant Recommendation System - Flask App
# Updated: April 2026

# Importing libraries
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Initialize flask app
app = Flask(__name__)

# Loading the datasets
zomato_df = pd.read_csv('restaurant1.csv')
df_percent = pd.read_csv('restaurant_percent.csv', index_col='name')

# Loading saved model files
cosine_similarities = pickle.load(
    open('cosine_similarities.pkl', 'rb')
)
indices = pickle.load(
    open('indices.pkl', 'rb')
)

# Recommendation function
def recommend(name, cosine_similarities=cosine_similarities):
    
    recommend_restaurant = []
    
    # Find index of restaurant
    idx = indices[indices == name].index[0]
    
    # Sort by cosine similarity
    score_series = pd.Series(
        cosine_similarities[idx]
    ).sort_values(ascending=False)
    
    # Get top 30 indexes
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Get restaurant names
    for each in top30_indexes:
        recommend_restaurant.append(
            list(df_percent.index)[each]
        )
    
    # Create result dataframe
    df_new = pd.DataFrame(
        columns=['cuisines', 'Mean Rating', 'cost']
    )
    
    for each in recommend_restaurant:
        df_new = df_new._append(
            pd.DataFrame(
                df_percent[['cuisines', 'Mean Rating', 'cost']][
                    df_percent.index == each
                ].sample()
            )
        )
    
    # Drop duplicates and sort by rating
    df_new = df_new.drop_duplicates(
        subset=['cuisines', 'Mean Rating', 'cost'],
        keep=False
    )
    df_new = df_new.sort_values(
        by='Mean Rating',
        ascending=False
    ).head(10)
    
    return df_new

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for extractor page
@app.route('/extractor')
def extractor():
    return render_template('extractor.html')

# Route for recommendation results
@app.route('/keywords', methods=['POST'])
def keywords():
    # Get restaurant name from form
    output = request.form.get('output')
    
    try:
        # Get recommendations
        result = recommend(output)
        result = result.to_string(index=False)
        
        # Convert to html table
        res = recommend(output)
        res = res.to_html()
        
    except:
        res = '''
        <div style="color:red; font-size:18px; margin:20px 0;">
            ❌ Restaurant not found! 
            Please try another restaurant name.
        </div>
        '''
    
    return render_template(
        'keywords.html', 
        keyword=res
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=False)