import streamlit as st
import praw
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from fuzzywuzzy import fuzz
import logging
from streamlit_autorefresh import st_autorefresh
import base64
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import os


# Load environment variables


# Auto refresh every 5 minutes
count = st_autorefresh(interval=300000, key="datarefresher")

# Setup logging
log_file = "reddit_analyzer.log"
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('reddit_analyzer')

# Directly define Reddit API Credentials
REDDIT_CREDENTIALS = {
    'client_id': 'eXzS0WLU_9tV1gq9p9H0Pg',
    'client_secret': 'waCj59cgW0Fz0fq8zPuoTK8nzRTzUQ',
    'user_agent': 'python:com.example.redditanalyzer:v1.0 (by /u/Training_Student1147)',
    'username': 'Training_Student1147',
    'password': 'Palu@#$123'
}

def setup_reddit_api():
    """Initialize and return Reddit API instance"""
    try:
        reddit_instance = praw.Reddit(
            client_id=REDDIT_CREDENTIALS['client_id'],
            client_secret=REDDIT_CREDENTIALS['client_secret'],
            user_agent=REDDIT_CREDENTIALS['user_agent'],
            username=REDDIT_CREDENTIALS['username'],
            password=REDDIT_CREDENTIALS['password'],
            check_for_async=False
        )
        return reddit_instance
    except Exception as e:
        logger.error(f"Failed to initialize Reddit API: {e}")
        st.error(f"Failed to initialize Reddit API: {e}")
        raise

# Create global reddit instance
try:
    reddit = setup_reddit_api()
except Exception as e:
    st.error(f"Failed to initialize Reddit API: {e}")
    reddit = None

def check_reddit_connection():
    """Test Reddit API connection"""
    if reddit is None:
        st.error("Reddit API not initialized")
        return False
        
    try:
        # Test the connection
        reddit.auth.limits
        logger.info("Reddit API connection successful")
        st.success("✅ Reddit API connection successful!")
        return True
    except Exception as e:
        logger.error(f"Reddit API connection failed: {e}")
        st.error(f"❌ Reddit API connection failed: {e}")
        return False

class AdvancedRedditDataCollector:
    def __init__(self):
        self.reddit = reddit
        self.cache = {}
        
    def parse_search_query(self, query):
        """Parse complex search queries with AND/OR/NOT operators"""
        terms = query.lower().split()
        must_include = set()
        must_exclude = set()
        
        i = 0
        while i < len(terms):
            if terms[i] == 'and':
                i += 1
                continue
            elif terms[i] == 'or':
                i += 1
                continue
            elif terms[i] == 'not':
                i += 1
                if i < len(terms):
                    must_exclude.add(terms[i])
            else:
                must_include.add(terms[i])
            i += 1
            
        return must_include, must_exclude

    def collect_data_api(self, subreddit, keywords, limit=100, title_only=False, sort_by='hot'):
        try:
            must_include, must_exclude = self.parse_search_query(" ".join(keywords))
            
            posts = []
            subreddit_obj = self.reddit.subreddit(subreddit)
            
            with st.spinner("Fetching posts..."):
                progress_bar = st.progress(0)
                
                # Get posts based on sort method
                if sort_by == 'hot':
                    submissions = subreddit_obj.hot(limit=limit)
                elif sort_by == 'new':
                    submissions = subreddit_obj.new(limit=limit)
                elif sort_by == 'top':
                    submissions = subreddit_obj.top(limit=limit)
                else:
                    submissions = subreddit_obj.hot(limit=limit)
                
                for idx, post in enumerate(submissions):
                    progress_bar.progress((idx + 1) / limit)
                    
                    # Search in title only or both title and body
                    search_text = post.title.lower() if title_only else f"{post.title.lower()} {post.selftext.lower()}"
                    
                    # Check must_include and must_exclude terms
                    if all(term in search_text for term in must_include) and \
                       not any(term in search_text for term in must_exclude):
                        
                        sentiment = TextBlob(post.title).sentiment
                        post_data = {
                            'title': post.title,
                            'score': post.score,
                            'url': post.url,
                            'author': str(post.author),
                            'created_utc': datetime.fromtimestamp(post.created_utc),
                            'sentiment_polarity': sentiment.polarity,
                            'sentiment_subjectivity': sentiment.subjectivity,
                            'num_comments': post.num_comments,
                            'upvote_ratio': post.upvote_ratio,
                            'text_preview': post.selftext[:200] + '...' if len(post.selftext) > 200 else post.selftext
                        }
                        posts.append(post_data)
                
                progress_bar.empty()
                
                if not posts:
                    st.warning("No matching posts found.")
                    return pd.DataFrame()
                    
                df = pd.DataFrame(posts)
                return df
                
        except Exception as e:
            logger.error(f"Error collecting data from API: {str(e)}")
            st.error(f"Error collecting data from API: {str(e)}")
            return None

    def generate_advanced_visualizations(self, df):
        """Generate advanced visualizations"""
        st.write("### 📊 Advanced Visualizations")
        
        # Time-based heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            x=df['created_utc'].dt.hour,
            y=df['created_utc'].dt.day_name(),
            z=df['score'],
            colorscale='Viridis'
        ))
        fig_heatmap.update_layout(
            title='Post Activity Heatmap',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week'
        )
        st.plotly_chart(fig_heatmap)
        
        # Sentiment over time
        fig_sentiment = px.scatter(df,
            x='created_utc',
            y='sentiment_polarity',
            size='score',
            color='num_comments',
            hover_data=['title'],
            title='Sentiment and Engagement Over Time'
        )
        st.plotly_chart(fig_sentiment)
        
        # Word cloud of titles
        if len(df) > 0:
            text = ' '.join(df['title'].astype(str))
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white'
            ).generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

def download_csv(df):
    """
    Function to generate a downloadable CSV file from the DataFrame.
    """
    if df is None or df.empty:
        st.error("No data available to export. Please perform an analysis first.")
        return False

    try:
        # Create a buffer for the CSV
        buffer = BytesIO()

        # Add timestamp to the file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"reddit_analysis_{timestamp}.csv"

        # Write DataFrame to the buffer
        df.to_csv(buffer, index=False, encoding='utf-8-sig')
        buffer.seek(0)

        # Streamlit download button
        st.download_button(
            label="📥 Download CSV File",
            data=buffer.getvalue(),  # Get CSV data from buffer
            file_name=filename,
            mime="text/csv",
            help="Click to download the analysis results as a CSV file."
        )
        return True

    except Exception as e:
        logger.error(f"Error while generating CSV: {str(e)}")
        st.error(f"Error while generating CSV: {str(e)}")

def save_data(df, filename):
    """Save DataFrame to a CSV file."""
    try:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        logger.info(f"Data saved successfully to {filename}")
        st.success(f"Data saved successfully to {filename}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        st.error(f"Error saving data: {str(e)}")

# Streamlit UI
st.title("Reddit Data Analyzer")

# Input for subreddit and keywords
subreddit = st.text_input("Enter Subreddit:", "learnpython")
keywords = st.text_input("Enter Keywords (space-separated):", "python programming")

# Button to fetch data
if st.button("Fetch Data"):
    if check_reddit_connection():
        collector = AdvancedRedditDataCollector()
        df = collector.collect_data_api(subreddit, keywords.split())
        
        if df is not None and not df.empty:
            st.write("### Data Preview")
            st.dataframe(df)

            # Generate visualizations
            collector.generate_advanced_visualizations(df)

            # Download button for CSV
            download_csv(df)
        else:
            st.warning("No data found for the given subreddit and keywords.")
