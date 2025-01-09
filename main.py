import streamlit as st
import praw
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import logging
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)

# Load environment variables

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
    reddit_instance = None
    try:
        reddit_instance = praw.Reddit(
            client_id=REDDIT_CREDENTIALS['client_id'],
            client_secret=REDDIT_CREDENTIALS['client_secret'],
            user_agent=REDDIT_CREDENTIALS['user_agent'],
            username=REDDIT_CREDENTIALS['username'],
            password=REDDIT_CREDENTIALS['password'],
            check_for_async=False
        )
        logger.info("Reddit API initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Reddit API: {e}")
        st.error(f"Failed to initialize Reddit API: {e}")
    return reddit_instance

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
        self.cache_expiration = timedelta(minutes=10)  # Cache expiration time
        self.cache = {}  # Cache to store API results
        self.api_call_count = {}  # Track API call counts for each query
        self.max_api_calls = 60  # Example limit for API calls

    def get_cache_key(self, subreddit, keywords):
        """Generate a unique cache key based on subreddit and keywords."""
        return f"{subreddit}:{' '.join(keywords)}"

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
        logger.info(f"Collecting data from subreddit: {subreddit} with keywords: {keywords}")
        try:
            current_time = datetime.now()  # Define current time
            must_include, must_exclude = self.parse_search_query(" ".join(keywords))
            
            # Create a cache key based on subreddit and keywords
            cache_key = self.get_cache_key(subreddit, keywords)

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
                
                # Log the columns of the DataFrame for debugging
                logger.info(f"DataFrame columns: {df.columns.tolist()}")
                
                # Check if the data is in the cache and not expired
                if cache_key in self.cache:
                    cached_data, timestamp = self.cache[cache_key]
                    if current_time - timestamp < self.cache_expiration:
                        logger.info("Returning cached data.")
                        return cached_data
                
                # Store the result in the cache
                self.cache[cache_key] = (df, datetime.now())
                
                # Update API call count only if a new API call is made
                if cache_key not in self.api_call_count:
                    self.api_call_count[cache_key] = 0
                self.api_call_count[cache_key] += 1
                
                # Print API call information
                remaining_calls = self.max_api_calls - self.api_call_count[cache_key]
                if remaining_calls < 0:
                    remaining_calls = 0  # Ensure remaining calls do not go negative
                logger.info(f"API calls for '{cache_key}': {self.api_call_count[cache_key]}, Remaining calls: {remaining_calls}")
                print(f"API calls for '{cache_key}': {self.api_call_count[cache_key]}, Remaining calls: {remaining_calls}")
                
                return df
                
        except Exception as e:
            logger.error(f"Error collecting data from API: {str(e)}")
            st.error(f"Error collecting data from API: {str(e)}")
            return None
def analyze_trending_topics(df):
    """Analyze subreddit data to identify trending topics."""
    if df.empty:
        return []
    topics = df['title'].value_counts().head(10)
    return topics

def date_wise_analysis(df):
    """Provide a timeline visualization of discussions."""
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df.set_index('created_utc', inplace=True)
    df.resample('M').size().plot()
    plt.title('Discussions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Discussions')
    st.pyplot(plt)

def audience_engagement_metrics(df):
    """Display audience engagement metrics."""
    total_users = df['author'].nunique()
    avg_sentiment = df['sentiment_polarity'].mean()
    st.write(f'Total Users Participating: {total_users}')
    st.write(f'Average Sentiment: {avg_sentiment:.2f}')

def generate_summary(df):
    """Generate a summary of discussions."""
    sentences = df['title'].apply(lambda x: TextBlob(x).sentences)
    summary = ' '.join([' '.join(map(str, sentence)) for sentence in sentences])
    return summary

def future_predictions(df):
    """Predict future relevance based on trends."""
    if df.empty:
        st.write("No data available for predictions.")
        return

    # Check if 'created_utc' column exists
    if 'created_utc' not in df.columns:
        st.write("The 'created_utc' column is missing from the data.")
        return

    # Example analysis: Calculate average sentiment and trends
    avg_sentiment = df['sentiment_polarity'].mean()
    st.write(f"Average Sentiment: {avg_sentiment:.2f}")

    # Trend analysis: Count posts over time
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    trend_data = df.set_index('created_utc').resample('M').size()
    
    # Plotting the trend
    fig = px.line(trend_data, x=trend_data.index, y=trend_data.values, title='Post Trends Over Time')
    st.plotly_chart(fig)

def suggest_related_subreddits(topic):
    """Suggest related subreddits based on the topic."""
    try:
        subreddit_results = reddit.subreddits.search_by_name(topic, exact=False)
        return [sub.display_name for sub in subreddit_results]
    except Exception as e:
        logger.error(f"Error fetching related subreddits: {str(e)}")
        return []

def fetch_and_display_posts(selected_subreddits, keywords):
    """Fetch and display posts based on selected subreddits and keywords."""
    collector = AdvancedRedditDataCollector()
    all_dataframes = []  # List to hold DataFrames for each selected subreddit
    
    for subreddit in selected_subreddits:
        try:
            df = collector.collect_data_api(subreddit, keywords)
            if df is not None and not df.empty:
                all_dataframes.append(df)
        except Exception as e:
            logger.error(f"Error collecting data for subreddit {subreddit}: {str(e)}")
            st.error(f"Error collecting data for subreddit {subreddit}: {str(e)}")

    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        st.write("### Combined Data Preview")
        st.dataframe(combined_df)

        # Display trending topics
        trending_topics = analyze_trending_topics(combined_df)
        st.write("### Trending Topics")
        st.bar_chart(trending_topics)

        # Date-wise analysis
        date_wise_analysis(combined_df)

        # Audience engagement metrics
        audience_engagement_metrics(combined_df)

        # Summary and insights
        summary = generate_summary(combined_df)
        st.write("### Summary of Discussions")
        st.write(summary)

        # Future predictions
        future_predictions(combined_df)

        # Download button for CSV
        download_csv(combined_df)
    else:
        st.warning("No data found for the selected subreddits and keywords.")

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

def keyword_frequency_analysis(df, keywords):
    """Analyze keyword frequency in the DataFrame."""
    keyword_counts = {keyword: 0 for keyword in keywords}
    
    for index, row in df.iterrows():
        title = row['title'].lower()
        for keyword in keywords:
            keyword_counts[keyword] += title.count(keyword.lower())
    
    # Create a bar chart for keyword frequencies
    st.write("### Keyword Frequency Analysis")
    st.bar_chart(keyword_counts)
    
# Streamlit UI
st.title("Reddit Data Analyzer")

# Input for subreddit and keywords
topic = st.text_input("Enter Topic:", "learnpython")
# Use Streamlit's session state to retain selected subreddits
if 'subreddits' not in st.session_state:
    st.session_state.subreddits = []  # Initialize session state for subreddits

# Button to fetch related subreddits
if st.button("Fetch Related Subreddits"):
    if check_reddit_connection():
        try:
            # Search for related subreddits based on the topic
            st.session_state.subreddits = suggest_related_subreddits(topic)  # Store in session state
            if st.session_state.subreddits:
                st.success("Related subreddits found!")
            else:
                st.warning("No related subreddits found.")
        except Exception as e:
            logger.error(f"Error fetching subreddits: {str(e)}")
            st.error(f"Error fetching subreddits: {str(e)}")

# Multi-select for user to choose subreddits
selected_subreddits = st.multiselect("Select Subreddits:", st.session_state.subreddits)

# Second input field for keywords
keyword_input = st.text_input("Enter Keywords (space-separated):", "")

# Button to fetch data
if st.button("Fetch Data"):
    if check_reddit_connection():
        fetch_and_display_posts(selected_subreddits, keyword_input.split())
