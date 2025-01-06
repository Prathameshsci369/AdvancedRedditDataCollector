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

# Auto refresh every 5 minutes
count = st_autorefresh(interval=300000, key="datarefresher")

# Setup logging
log_file = "reddit_analyzer.log"
logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('reddit_analyzer')

# Reddit API Credentials (directly in code)
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
        return False


def export_data(df):
    """Enhanced data export functionality"""
    st.write("### 💾 Export Options")
    
    with st.expander("Export Settings", expanded=True):
        # Column selection
        cols_to_export = st.multiselect(
            "Select columns to export",
            options=df.columns.tolist(),
            default=df.columns.tolist(),
            help="Choose which columns to include in the export"
        )
        
        # Format options
        col1, col2 = st.columns(2)
        with col1:
            include_metadata = st.checkbox(
                "Include metadata",
                value=True,
                help="Add analysis timestamp and parameters"
            )
        with col2:
            sort_by_col = st.selectbox(
                "Sort by",
                options=['None'] + df.columns.tolist(),
                help="Choose column to sort the data"
            )
    
        # Prepare export data
        export_df = df[cols_to_export].copy() if cols_to_export else df.copy()
        
        # Add metadata if selected
        if include_metadata:
            metadata = pd.DataFrame({
                'Analysis Info': [
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Total Posts: {len(df)}",
                    f"Subreddits: {', '.join(df['subreddit'].unique())}",
                    f"Average Score: {df['score'].mean():.2f}",
                    f"Average Comments: {df['num_comments'].mean():.2f}",
                    f"Average Sentiment: {df['sentiment_polarity'].mean():.2f}"
                ]
            })
            
            # Create a new buffer for combined data
            buffer = BytesIO()
            
            # Write metadata and data to buffer
            try:
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    metadata.to_excel(writer, sheet_name='Metadata', index=False)
                    export_df.to_excel(writer, sheet_name='Data', index=False)
                buffer.seek(0)
            except Exception as e:
                logger.error(f"Error writing to Excel buffer: {str(e)}")
                st.error(f"Error writing to Excel buffer: {str(e)}")
                return False
            
        # Sort if selected
        if sort_by_col != 'None':
            export_df = export_df.sort_values(by=sort_by_col, ascending=False)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            # CSV Download
            if st.button("📥 Download CSV", use_container_width=True):
                if not download_csv(export_df):
                    st.error("Failed to download CSV file.")
        
        with col2:
            # Excel Download
            if st.button("📊 Download Excel", use_container_width=True):
                try:
                    buffer = BytesIO()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"reddit_analysis_{timestamp}.xlsx"
                    
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        if include_metadata:
                            metadata.to_excel(writer, sheet_name='Metadata', index=False)
                        export_df.to_excel(writer, sheet_name='Data', index=False)
                    
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📊 Download Excel File",
                        data=buffer,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                except Exception as e:
                    logger.error(f"Error creating Excel file: {str(e)}")
                    st.error(f"Error creating Excel file: {str(e)}")

def show_combined_analysis(df):
    """Show analysis with enhanced export options"""
    st.write("## 📊 Analysis Results")
    
    # Quick stats
    with st.expander("📈 Quick Statistics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Posts", len(df))
        with col2:
            st.metric("Avg. Score", f"{df['score'].mean():.1f}")
        with col3:
            st.metric("Avg. Comments", f"{df['num_comments'].mean():.1f}")
        with col4:
            st.metric("Avg. Sentiment", f"{df['sentiment_polarity'].mean():.2f}")
    
    # Results table with filtering
    st.write("### 📋 Detailed Results")
    
    # Column visibility toggle
    with st.expander("🔍 Column Settings"):
        visible_cols = st.multiselect(
            "Select columns to display",
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )
    
    # Show filtered dataframe
    if visible_cols:
        st.dataframe(
            df[visible_cols],
            use_container_width=True,
            height=400
        )
    
    # Export options
    export_data(df)

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_posts(subreddit, limit, sort_by):
    """Fetch posts with caching and batching"""
    try:
        posts = []
        batch_size = 100  # Reddit's optimal batch size
        
        for i in range(0, limit, batch_size):
            current_batch = min(batch_size, limit - i)
            
            # Use PRAW's stream for efficient fetching
            if sort_by == 'hot':
                batch = reddit.subreddit(subreddit).hot(limit=current_batch)
            elif sort_by == 'new':
                batch = reddit.subreddit(subreddit).new(limit=current_batch)
            else:
                batch = reddit.subreddit(subreddit).top(limit=current_batch)
                
            posts.extend(list(batch))
            
        return posts
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")
        st.error(f"Error fetching posts: {e}")
        return []

def show_range_explanations():
    """Show explanations for different ranges"""
    with st.expander("ℹ️ Understanding Scores and Sentiment"):
        st.markdown("""
        ### 📊 Score Range
        - **What is it?** Post upvotes minus downvotes
        - **Higher scores** = More popular posts
        - **Lower scores** = Less popular or controversial posts
        
        ### 😊 Sentiment Range
        - **Range**: -1.0 to +1.0
        - **Negative** (-1.0 to 0): Negative emotion/opinion
        - **Neutral** (around 0): Neutral or factual content
        - **Positive** (0 to +1.0): Positive emotion/opinion
        
        Examples:
        - "I love this!" = +0.8
        - "This is okay" = +0.2
        - "This doesn't work" = -0.3
        - "This is terrible" = -0.8
        """)

# Mobile-friendly layout adjustments
def create_mobile_friendly_layout():
    """Create responsive layout"""
    # Check if mobile
    is_mobile = st.session_state.get('is_mobile', False)
    
    if is_mobile:
        # Single column layout
        st.write("### 📱 Reddit Analyzer")
        subreddit = st.text_input("Subreddit:")
        limit = st.slider("Posts:", 10, 2000, 100)
    else:
        # Desktop layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 💻 Reddit Analyzer")
            subreddit = st.text_input("Subreddit:")
        with col2:
            limit = st.number_input("Number of posts:", 10, 2000, 100)

class OptimizedRedditCollector:
    def __init__(self):
        self.cache = {}
        self.last_request_time = {}
        
    @st.cache_data(ttl=3600)
    def get_cached_posts(self, subreddit, limit):
        """Get posts with caching"""
        cache_key = f"{subreddit}_{limit}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        posts = self.fetch_posts_with_backoff(subreddit, limit)
        self.cache[cache_key] = posts
        return posts
        
    def fetch_posts_with_backoff(self, subreddit, limit):
        """Fetch posts with exponential backoff"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Check rate limits
                current_time = time.time()
                if subreddit in self.last_request_time:
                    time_diff = current_time - self.last_request_time[subreddit]
                    if time_diff < 2:  # Minimum 2 seconds between requests
                        time.sleep(2 - time_diff)
                
                posts = fetch_posts(subreddit, limit)
                self.last_request_time[subreddit] = time.time()
                return posts
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(retry_delay)
                retry_delay *= 2

class SubredditFinder:
    def __init__(self):
        self.reddit = reddit
        
    def find_similar_subreddits(self, keyword, limit=10):
        """Find similar subreddits based on keyword"""
        try:
            similar_subreddits = []
            
            # Cache key for this search
            cache_key = f"subreddit_search_{keyword}_{limit}"
            
            # Check if results are in session state
            if cache_key in st.session_state:
                return st.session_state[cache_key]
            
            # Search for subreddits
            with st.spinner("🔍 Searching for relevant subreddits..."):
                # Direct search
                for subreddit in self.reddit.subreddits.search(keyword, limit=limit):
                    sub_info = {
                        'name': subreddit.display_name,
                        'title': subreddit.title,
                        'description': subreddit.public_description[:200] + '...' if len(subreddit.public_description) > 200 else subreddit.public_description,
                        'subscribers': subreddit.subscribers,
                        'relevance_score': self._calculate_relevance(keyword, subreddit)
                    }
                    similar_subreddits.append(sub_info)
                
                # Sort by relevance
                similar_subreddits.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                # Cache the results
                st.session_state[cache_key] = similar_subreddits
                
            return similar_subreddits
            
        except Exception as e:
            logger.error(f"Error finding similar subreddits: {e}")
            st.error(f"Error finding similar subreddits: {e}")
            return []
    
    def _calculate_relevance(self, keyword, subreddit):
        """Calculate relevance score for a subreddit"""
        keyword = keyword.lower()
        name_match = fuzz.ratio(keyword, subreddit.display_name.lower())
        title_match = fuzz.ratio(keyword, subreddit.title.lower())
        desc_match = fuzz.ratio(keyword, subreddit.public_description.lower())
        
        # Weighted score
        return (name_match * 0.5 + title_match * 0.3 + desc_match * 0.2)

@st.cache_data(ttl=3600)
def search_subreddits(keyword, limit=10):
    """Cached wrapper for subreddit search"""
    finder = SubredditFinder()
    return finder.find_similar_subreddits(keyword, limit)

def show_subreddit_selector(keyword):
    """Show subreddit selection interface with multiple selection"""
    similar_subs = search_subreddits(keyword)
    
    if not similar_subs:
        st.warning("No relevant subreddits found.")
        return None
        
    st.write("### 🎯 Relevant Subreddits Found")
    st.info("Select one or more subreddits to analyze:")
    
    # Store selected subreddits
    if 'selected_subreddits' not in st.session_state:
        st.session_state.selected_subreddits = set()
    
    # Display subreddits in a grid
    cols = st.columns(2)
    selected_any = False
    
    for idx, sub in enumerate(similar_subs):
        with cols[idx % 2]:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"### r/{sub['name']}")
                    st.write(f"*{sub['title']}*")
                    st.write(f"👥 {sub['subscribers']:,} members")
                with col2:
                    # Checkbox for selection
                    is_selected = st.checkbox(
                        "Select",
                        key=f"select_{sub['name']}",
                        value=sub['name'] in st.session_state.selected_subreddits
                    )
                    if is_selected:
                        st.session_state.selected_subreddits.add(sub['name'])
                        selected_any = True
                    elif sub['name'] in st.session_state.selected_subreddits:
                        st.session_state.selected_subreddits.remove(sub['name'])
                
                st.write(sub['description'])
                st.write("---")
    
    # Return selected subreddits if any
    return list(st.session_state.selected_subreddits) if selected_any else None

def analyze_multiple_subreddits(subreddits, keywords, limit, title_only, sort_by):
    """Analyze multiple subreddits and combine results"""
    all_results = []
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        for idx, subreddit in enumerate(subreddits):
            progress_text.write(f"Analyzing r/{subreddit}... ({idx + 1}/{len(subreddits)})")
            progress_bar.progress((idx + 1) / len(subreddits))
            
            collector = AdvancedRedditDataCollector()
            df = collector.collect_data_api(
                subreddit,
                keywords,
                limit=limit,
                title_only=title_only,
                sort_by=sort_by
            )
            
            if df is not None and not df.empty:
                df['subreddit'] = subreddit  # Add subreddit column
                all_results.append(df)
            else:
                st.warning(f"No results found for r/{subreddit}")
        
        progress_text.empty()
        progress_bar.empty()
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return None
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        st.error(f"Error during analysis: {str(e)}")
        return None

def main():
    try:
        # Sidebar configuration
        st.sidebar.title("🔧 Configuration")

        # Basic Settings in Sidebar
        st.sidebar.header("Basic Settings")
        sort_by = st.sidebar.selectbox(
            "Sort posts by",
            ['hot', 'new', 'top'],
            help="Choose how to sort the Reddit posts"
        )

        # Main content area
        st.title("Advanced Reddit Data Analyzer")

        # Topic/Keyword Input
        st.write("### 🔍 Enter Your Topic")
        topic = st.text_input(
            "Enter a topic or keyword (e.g., 'JEE Mains', 'NEET', 'Engineering')",
            help="We'll find relevant subreddits for your topic"
        )

        if topic:
            # Find and select subreddits
            selected_subreddits = show_subreddit_selector(topic)

            if selected_subreddits:
                st.success(f"Selected subreddits: {', '.join(['r/' + sub for sub in selected_subreddits])}")

                # Analysis options
                with st.expander("🔧 Analysis Options", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        posts_per_sub = st.number_input(
                            "Posts per subreddit",
                            min_value=10,
                            max_value=1000,
                            value=100,
                            step=10
                        )
                    with col2:
                        title_only = st.checkbox(
                            "Search in titles only",
                            help="If checked, will only search post titles"
                        )

                # Search terms
                st.write("### 🔎 Search Terms")
                default_terms = topic.split()
                keywords = st.text_area(
                    "Enter search terms (one per line)",
                    value="\n".join(default_terms),
                    height=100,
                    help="Enter each search term on a new line"
                ).split('\n')
                keywords = [k.strip() for k in keywords if k.strip()]

                # Analysis button
                if st.button("🚀 Start Analysis", use_container_width=True):
                    if not keywords:
                        st.error("Please enter search terms.")
                    else:
                        try:
                            with st.spinner("Analyzing selected subreddits..."):
                                collector = AdvancedRedditDataCollector()
                                df = analyze_multiple_subreddits(
                                    selected_subreddits,
                                    keywords,
                                    posts_per_sub,
                                    title_only,
                                    sort_by
                                )
                                if df is not None and not df.empty:
                                    # Save results to session state
                                    st.session_state.current_df = df
                                    st.session_state.filtered_df = df

                                    # Show combined analysis
                                    show_combined_analysis(df)

                                    # Call visualization functions
                                    collector.generate_advanced_visualizations(df)
                                else:
                                    st.warning("No matching posts found in any selected subreddit.")
                        except Exception as e:
                            st.error(f"An error occurred during analysis: {str(e)}")
                            logger.error(f"Analysis error: {str(e)}")

        else:
            st.info("Enter a topic to begin the analysis.")

    except Exception as main_error:
        st.error(f"An unexpected error occurred: {str(main_error)}")
        logger.error(f"Main function error: {str(main_error)}")



def show_sidebar_filters(df):
    """Show filters in sidebar"""
    st.sidebar.header("Filter Settings")
    
    # Score filter
    max_score = max(int(df['score'].max()), 1)
    score_range = st.sidebar.slider(
        "Score range",
        min_value=0,
        max_value=max_score,
        value=(0, max_score),
        help="Filter posts by their score (upvotes)"
    )
    
    # Comment filter
    max_comments = max(int(df['num_comments'].max()), 1)
    comment_range = st.sidebar.slider(
        "Comment range",
        min_value=0,
        max_value=max_comments,
        value=(0, max_comments),
        help="Filter posts by number of comments"
    )
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date range",
        value=(
            df['created_utc'].min().date(),
            df['created_utc'].max().date()
        ),
        help="Filter posts by date"
    )
    
    # Apply filters
    if st.sidebar.button("🔄 Apply Filters", use_container_width=True):
        filtered_df = df[
            (df['score'].between(score_range[0], score_range[1])) &
            (df['num_comments'].between(comment_range[0], comment_range[1])) &
            (df['created_utc'].dt.date.between(date_range[0], date_range[1]))
        ]
        st.session_state.filtered_df = filtered_df

def show_combined_analysis(df):
    """Show analysis results with enhanced exception handling."""
    try:
        st.write("## 📊 Analysis Results")

        # Quick stats
        with st.expander("📈 Quick Statistics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Posts", len(df))
            with col2:
                st.metric("Avg. Score", f"{df['score'].mean():.1f}")
            with col3:
                st.metric("Avg. Comments", f"{df['num_comments'].mean():.1f}")
            with col4:
                st.metric("Avg. Sentiment", f"{df['sentiment_polarity'].mean():.2f}")

        # Results table
        st.write("### 📋 Detailed Results")
        st.dataframe(df, use_container_width=True)

        # Download options
        st.write("### 💾 Download Results")
        if st.button("📥 Download CSV"):
            download_csv(df)

    except Exception as e:
        st.error(f"An error occurred while displaying results: {str(e)}")
        logger.error(f"Error in show_combined_analysis: {str(e)}")


if __name__ == "__main__":
    main()
