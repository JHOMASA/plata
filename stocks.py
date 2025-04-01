import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from textblob import TextBlob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Configuration
FINGPT_API_KEY = "AIzaTRDjNFU6WAx6FJ74zhm2vQqWyD5MsYKUcOk"  # Replace with actual key
NEWS_API_KEY = "3f8e6bb1fb72490b835c800afcadd1aa"      # Replace with actual key

def fetch_stock_data(ticker, period='1y'):
    data = yf.download(ticker, period=period)
    if data.empty:
        dates = pd.date_range(end=datetime.today(), periods=100)
        data = pd.DataFrame({
            'Close': np.random.normal(100, 10, 100).cumsum(),
            'Volume': np.random.randint(100000, 1000000, 100),
            'Open': np.random.normal(100, 10, 100),
            'High': np.random.normal(105, 10, 100),
            'Low': np.random.normal(95, 10, 100)
        }, index=dates)
    return data

# ======================
# PREDICTION MODELS
# ======================
def train_holt_winters(data, seasonal_periods=5, damped=True):
    model = ExponentialSmoothing(
        data['Close'],
        trend='add',
        seasonal='add',
        seasonal_periods=seasonal_periods,
        damped_trend=damped
    ).fit()
    return model

def predict_holt_winters(model, periods=30):
    return model.forecast(periods).values

def train_prophet_model(data):
    df = data.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(daily_seasonality=False)
    model.fit(df)
    return model

def train_lstm_model(data, lookback=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)
    return model, scaler


# Enhanced FinGPTChat Class with Visualization
class FinGPTChat:
    def __init__(self, section_name, context="", visual_data=None):
        self.section = section_name
        self.context = context
        self.visual_data = visual_data or {}
        self.messages = [
            {
                "role": "system",
                "content": f"""You are a {self._get_section_expertise()} analyst. Provide:
1. References to existing visualizations (mention "as shown above")
2. NEW visualization suggestions when helpful
3. Bullet points for key insights
4. Risk assessment and alternatives
5. Current context: {context}"""
            }
        ]
    
    def _get_section_expertise(self):
        expertise_map = {
            "stock": "technical and fundamental equity",
            "monte_carlo": "quantitative risk and probability",
            "ratios": "financial health and benchmarking",
            "sentiment": "behavioral finance and market psychology",
            "predictions": "algorithmic trading strategy",
            "recommendations": "portfolio management",
            "news": "market-moving event analysis"
        }
        return expertise_map.get(self.section, "financial")
    
    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            
            # Generate the textual response
            text_response = self._get_text_response(question)
            
            # Generate visualization suggestions
            viz_components = self._generate_visualizations(question, text_response)
            
            return text_response, viz_components
            
        except Exception as e:
            return f"Error: {str(e)}", []

    def _get_text_response(self, question):
        """Simulates FinGPT responses - replace with actual API calls"""
        if self.section == "stock":
            return self._stock_response(question)
        elif self.section == "monte_carlo":
            return self._monte_carlo_response(question)
        elif self.section == "ratios":
            return self._ratios_response(question)
        elif self.section == "sentiment":
            return self._sentiment_response(question)
        elif self.section == "recommendations":
            return self._recommendation_response(question)
        elif self.section == "news":
            return self._news_response(question)
        else:
            return self._prediction_response(question)
    
    def _generate_visualizations(self, question, response):
        """Generates visualization suggestions based on question"""
        suggestions = []
        
        if "correlation" in question.lower():
            suggestions.append(("correlation", self._plot_correlation()))
        
        if "seasonality" in question.lower():
            suggestions.append(("seasonality", self._plot_seasonality()))
            
        if "probability" in question.lower() and self.section == "monte_carlo":
            suggestions.append(("probability_dist", self._plot_probability_dist()))
            
        if "sentiment trend" in question.lower():
            suggestions.append(("sentiment_timeline", self._plot_sentiment_timeline()))
            
        if "price targets" in question.lower() and self.section == "predictions":
            suggestions.append(("price_targets", self._plot_price_targets()))
            
        return suggestions
    
    # Visualization generation methods
    def _plot_correlation(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        # Mock correlation matrix
        ax.matshow(np.random.rand(5,5), cmap='coolwarm')
        ax.set_title("Asset Correlation Matrix")
        return self._fig_to_base64(fig)
    
    def _plot_probability_dist(self):
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=np.random.normal(100, 15, 1000),
            name="Price Distribution"
        ))
        return fig.to_image(format="png")
    
    # Section-specific response templates
    def _stock_response(self, question):
        return f"""Based on the {self.visual_data.get('period', '1-year')} chart:

• Current Trend: {self.visual_data.get('trend', 'upward')} 
• Key Levels: 
  - Support: ${self.visual_data.get('support', 'N/A')} 
  - Resistance: ${self.visual_data.get('resistance', 'N/A')}
• Volume: {'increasing' if self.visual_data.get('volume_up', False) else 'stable'}

Recommendation: {self._generate_recommendation()}"""

    def _monte_carlo_response(self, question):
        return f"""Risk Analysis:

• {self.visual_data.get('simulations', 1000)} simulations run
• Probability of 10% gain: {self.visual_data.get('prob_gain', '25%')}
• Probability of 10% loss: {self.visual_data.get('prob_loss', '15%')}
• Worst-case (5th %ile): ${self.visual_data.get('worst_case', 'N/A')}

As shown in the distribution above, tail risks are {'moderate' if float(self.visual_data.get('prob_loss', '0.15').strip('%')) < 20 else 'high'}."""

    def _recommendation_response(self, question):
        return f"""Portfolio Strategy:

• Allocation Suggestion: {self.visual_data.get('allocation', '3-5% of portfolio')}
• Time Horizon: {self.visual_data.get('horizon', '6-12 months')}
• Risk-Adjusted Return: {self.visual_data.get('sharpe', '1.2')} (Good)
• Hedge Suggestions: {self.visual_data.get('hedge', 'Protective puts at 5% below current')}"""

    def _fig_to_base64(self, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

# Data Fetching and Model Functions (Keep your existing implementations)
# [Include all your existing functions like fetch_stock_data, generate_recommendations, 
# prepare_lstm_data, train_lstm_model, predict_lstm, etc.]

# Enhanced Section Chat Implementations
def stock_analysis_chat(stock_data, ticker):
    visual_data = {
        "period": f"{len(stock_data)}-day",
        "trend": "up" if stock_data['Close'].iloc[-1] > stock_data['Close'].iloc[-30] else "down",
        "support": stock_data['Close'].min(),
        "resistance": stock_data['Close'].max(),
        "volume_up": stock_data['Volume'].iloc[-1] > stock_data['Volume'].mean()
    }
    
    chat = FinGPTChat("stock", f"Analyzing {ticker}", visual_data)
    
    with st.expander("💬 Advanced Technical Analysis"):
        user_question = st.text_input("Ask about patterns or trading signals:", key="stock_q")
        
        if user_question:
            response, viz_suggestions = chat.ask(user_question)
            
            st.markdown(f"**FinGPT Analysis**")
            st.markdown(response)
            
            for viz_type, viz_data in viz_suggestions:
                if viz_type == "correlation":
                    st.image(f"data:image/png;base64,{viz_data}", 
                            caption="Asset Correlation Matrix")
                elif viz_type == "seasonality":
                    st.image(f"data:image/png;base64,{viz_data}", 
                            caption="Seasonal Pattern")

def monte_carlo_simulation(stock_data, num_simulations=1000, days=252, method='basic'):
    """
    Enhanced Monte Carlo simulation with multiple methods
    Options for method: 'basic', 'ets', 'stl'
    """
    try:
        if stock_data.empty:
            raise ValueError("No stock data available for simulation.")

        returns = stock_data['Close'].pct_change().dropna()
        if len(returns) < 2:
            raise ValueError("Insufficient data to calculate returns.")

        simulations = np.zeros((days, num_simulations))
        S0 = stock_data['Close'].iloc[-1]

        if method == 'basic':
            mu = returns.mean()
            sigma = returns.std()
            
            for i in range(num_simulations):
                daily_returns = np.random.normal(mu, sigma, days)
                simulations[:, i] = S0 * (1 + daily_returns).cumprod()

        elif method == 'ets':
            model = ExponentialSmoothing(stock_data['Close'], trend='add', damped_trend=True).fit()
            for i in range(num_simulations):
                sim = model.simulate(days, repetitions=1, error_variance=model.sse/len(stock_data))
                simulations[:, i] = sim.values.flatten()

        elif method == 'stl':
            stl = STL(stock_data['Close'], period=63).fit()
            resid_std = stl.resid.std()
            trend_component = np.linspace(stl.trend.iloc[-1],
                                          stl.trend.iloc[-1] + (stl.trend.iloc[-1] - stl.trend.iloc[-2]) * days,
                                          days)
            seasonal_component = np.tile(stl.seasonal[-63:], days // 63 + 1)[:days]
            
            for i in range(num_simulations):
                noise = np.random.normal(0, resid_std, days)
                simulations[:, i] = trend_component + seasonal_component + noise
        
        return simulations
    
    except Exception as e:
        st.error(f"Error in Monte Carlo simulation ({method}): {e}")
        return None

# Streamlit UI for Risk Scenario Analysis
st.sidebar.header("🔍 Risk Scenario Analysis")
q = st.sidebar.selectbox("Common risk questions:", [
    "What are the key risk probabilities?",
    "How should I interpret the worst-case scenario?",
    "Custom question..."
], key="mc_q")

if q == "Custom question...":
    q = st.sidebar.text_input("Enter your question:", key="mc_custom")

if q and q != "Custom question...":
    st.subheader("Risk Assessment")
    st.write("Response coming from AI model...")  # Placeholder for AI model integration
    
    # Monte Carlo Visualization
    terminal_prices = monte_carlo_simulation(pd.DataFrame({'Close': np.random.rand(500) * 100}))[-1, :]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=terminal_prices, name="Outcomes"))
    fig.update_layout(title="Terminal Price Distribution")
    st.plotly_chart(fig)

def financial_ratios_chat(ratios, ticker):
    chat = FinGPTChat("ratios", f"Ratios for {ticker}: {json.dumps(ratios)}")
    
    with st.expander("📊 Ratio Deep Dive"):
        question = st.selectbox("Ask about:", [
            "How do these compare to sector averages?",
            "Which ratios concern you most?",
            "What's the overall financial health?"
        ], key="ratio_q")
        
        if question:
            response, _ = chat.ask(question)
            st.markdown(f"**Ratio Analysis**")
            st.markdown(response)
            
            # Generate benchmark comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(ratios.keys()),
                y=[float(r.strip('%')) for r in ratios.values()],
                name="Current"
            ))
            # Add sector averages (mock data)
            fig.add_trace(go.Bar(
                x=list(ratios.keys()),
                y=[15, 20, 1.2, 5],  # Mock benchmarks
                name="Sector Avg"
            ))
            fig.update_layout(title="Ratio Benchmarking")
            st.plotly_chart(fig)

def news_sentiment_chat(articles, sentiment_counts, ticker):
    visual_data = {
        "positive": sentiment_counts.get("Positive", 0),
        "negative": sentiment_counts.get("Negative", 0),
        "sample_headlines": [a['title'][:50] + "..." for a in articles[:3]]
    }
    
    chat = FinGPTChat("sentiment", f"News for {ticker}", visual_data)
    
    with st.expander("📰 Sentiment Interpretation"):
        q = st.text_input("Ask about sentiment implications:", key="sentiment_q")
        
        if q:
            response, viz_suggestions = chat.ask(q)
            
            st.markdown(f"**Market Psychology Analysis**")
            st.markdown(response)
            
            # Sentiment timeline if requested
            if "sentiment_timeline" in [v[0] for v in viz_suggestions]:
                dates = [a['publishedAt'][:10] for a in articles if 'publishedAt' in a]
                sentiments = []
                for a in articles:
                    if 'publishedAt' in a:
                        blob = TextBlob(f"{a.get('title','')} {a.get('description','')}")
                        sentiments.append(blob.sentiment.polarity)
                
                if dates and sentiments:
                    df = pd.DataFrame({'date': pd.to_datetime(dates), 'sentiment': sentiments})
                    df = df.groupby('date').mean().reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['date'], y=df['sentiment'],
                        name="Sentiment Trend"
                    ))
                    fig.update_layout(title="Daily Sentiment Trend")
                    st.plotly_chart(fig)

def latest_news_chat(articles, ticker):
    chat = FinGPTChat("news", f"Latest news for {ticker}")
    
    with st.expander("🗞️ News Analysis"):
        q = st.selectbox("Ask about:", [
            "Which news items are most market-moving?",
            "Are there any emerging themes?",
            "Custom question..."
        ], key="news_q")
        
        if q == "Custom question...":
            q = st.text_input("Enter your question:", key="news_custom")
        
        if q and q != "Custom question...":
            response, _ = chat.ask(q)
            
            st.markdown(f"**News Impact Analysis**")
            st.markdown(response)
            
            # Generate news impact timeline
            if "impact" in q.lower():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime([a['publishedAt'][:10] for a in articles[:10]]),
                    y=[len(a['title']) for a in articles[:10]],  # Mock impact score
                    name="News Impact",
                    mode="markers",
                    marker=dict(size=12)
                ))
                fig.update_layout(title="Recent News Impact Scores")
                st.plotly_chart(fig)

def recommendations_chat(recommendations, ticker, financial_ratios):
    visual_data = {
        "allocation": "5-8%" if float(financial_ratios.get('Volatility', '0%').strip('%')) < 15 else "2-5%",
        "horizon": "3-6 months" if float(financial_ratios.get('Sharpe Ratio', 1)) > 1 else "6-12 months",
        "sharpe": financial_ratios.get('Sharpe Ratio', 'N/A'),
        "hedge": "10% OTM puts" if float(financial_ratios.get('Max Drawdown', '0%').strip('%')) > 15 else "5% OTM puts"
    }
    
    chat = FinGPTChat("recommendations", f"Recommendations for {ticker}", visual_data)
    
    with st.expander("📈 Portfolio Strategy Advisor"):
        q = st.selectbox("Common questions:", [
            "How should I size this position?",
            "What's the optimal time horizon?",
            "How should I hedge this investment?"
        ], key="rec_q")
        
        if q == "Custom question...":
            q = st.text_input("Enter your question:", key="rec_custom")
        
        if q and q != "Custom question...":
            response, _ = chat.ask(q)
            
            st.markdown(f"**Portfolio Strategy**")
            st.markdown(response)
            
            # Generate allocation pie chart
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=["Target Stock", "Hedge", "Cash"],
                values=[5, 2, 93],
                name="Suggested Allocation"
            ))
            fig.update_layout(title="Portfolio Allocation Strategy")
            st.plotly_chart(fig)

def predictions_chat(predictions, ticker, model_type, historical_data):
    visual_data = {
        "model": model_type,
        "predicted_change": f"{((predictions[-1]/historical_data['Close'].iloc[-1])-1)*100:.1f}%",
        "current_price": historical_data['Close'].iloc[-1],
        "confidence": "high" if model_type in ["LSTM","Prophet"] else "medium"
    }
    
    chat = FinGPTChat("predictions", f"{model_type} predictions for {ticker}", visual_data)
    
    with st.expander("🎯 Strategy Advisor"):
        q = st.selectbox("Strategy questions:", [
            "What's the recommended position size?",
            "Should I hedge this position?",
            "What price targets make sense?"
        ], key="pred_q")
        
        if q:
            response, viz_suggestions = chat.ask(q)
            
            st.markdown(f"**{model_type} Strategy**")
            st.markdown(response)
            
            # Add strategy backtest visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=historical_data.index[-30:],
                y=historical_data['Close'].values[-30:],
                name="Historical"
            ))
            future_dates = pd.date_range(
                start=historical_data.index[-1],
                periods=len(predictions)+1
            )[1:]
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                name="Predicted",
                line=dict(color='green', dash='dot')
            ))
            fig.update_layout(title="Prediction Backtesting")
            st.plotly_chart(fig)

# Main App with Complete FinGPT Integration
def main():
    st.title("Stock Analysis Dashboard")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Stock Analysis", "Monte Carlo Simulation", "Financial Ratios", 
               "News Sentiment", "Latest News", "Recommendations", "Predictions"]
    choice = st.sidebar.radio("Choose a section", options)

    if choice == "Stock Analysis":
        st.header("Stock Analysis")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                st.write("### Stock Data")
                st.write(stock_data)

                # Plot stock data
                st.write("### Stock Price Chart")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
                fig.update_layout(title=f"Stock Price for {stock_ticker}", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)
                
                # Add FinGPT chat
                stock_analysis_chat(stock_data, stock_ticker)

    elif choice == "Monte Carlo Simulation":
        st.header("Monte Carlo Simulation")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                simulations = monte_carlo_simulation(stock_data)
                if simulations is not None:
                    st.write("### Monte Carlo Simulation Results")
                    fig = go.Figure()
                    for i in range(min(10, simulations.shape[1])):
                        fig.add_trace(go.Scatter(
                            x=np.arange(simulations.shape[0]),
                            y=simulations[:, i],
                            mode='lines',
                            name=f'Simulation {i+1}'
                        ))
                    fig.update_layout(title="Monte Carlo Simulation", xaxis_title="Days", yaxis_title="Price")
                    st.plotly_chart(fig)
                    
                    # Add FinGPT chat
                    monte_carlo_chat(simulations, stock_ticker)

    elif choice == "Financial Ratios":
        st.header("Financial Ratios")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                risk_metrics = calculate_risk_metrics(stock_data)
                st.write("### Financial Ratios")
                st.table(pd.DataFrame(list(risk_metrics.items()), columns=["Ratio", "Value"]))
                
                # Add FinGPT chat
                financial_ratios_chat(risk_metrics, stock_ticker)

    elif choice == "News Sentiment":
        st.header("News Sentiment Analysis")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            articles = fetch_news(stock_ticker)
            if articles:
                sentiment_counts = analyze_news_sentiment(articles)
                st.write("### Sentiment Summary")
                st.write(f"Positive: {sentiment_counts['Positive']}")
                st.write(f"Negative: {sentiment_counts['Negative']}")
                st.write(f"Neutral: {sentiment_counts['Neutral']}")
                st.write(f"Errors: {sentiment_counts['Error']}")

                # Plot sentiment distribution
                st.write("### Sentiment Distribution")
                fig = go.Figure(data=[go.Bar(
                    x=list(sentiment_counts.keys()),
                    y=list(sentiment_counts.values())
                )])
                fig.update_layout(
                    title="News Sentiment Analysis",
                    xaxis_title="Sentiment",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig)
                
                # Add FinGPT chat
                news_sentiment_chat(articles, sentiment_counts, stock_ticker)
            else:
                st.warning("No news articles found for this stock ticker.")

    elif choice == "Latest News":
        st.header("Latest News")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        if st.button("Submit"):
            articles = fetch_news(stock_ticker)
            if articles:
                st.write("### Top 5 News Articles")
                for article in articles[:5]:
                    st.write(f"**Title:** {article.get('title', 'No Title Available')}")
                    st.write(f"**Description:** {article.get('description', 'No Description Available')}")
                    st.write(f"**Source:** {article.get('source', {}).get('name', 'N/A')}")
                    st.write(f"**Published At:** {article.get('publishedAt', 'N/A')}")
                    st.write("---")
                
                # Add FinGPT chat
                latest_news_chat(articles, stock_ticker)
            else:
                st.warning("No news articles found for this stock ticker.")

    elif choice == "Recommendations":
        st.header("Recommendations")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        period = st.number_input("Enter Analysis Period (days)", value=30)
        if st.button("Submit"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                financial_ratios = calculate_risk_metrics(stock_data)
                recommendations = generate_recommendations(stock_data, financial_ratios, period)
                st.write("### Recommendations")
                for recommendation in recommendations:
                    st.write(recommendation)
                
                # Add FinGPT chat
                recommendations_chat(recommendations, stock_ticker, financial_ratios)

    elif choice == "Predictions":
        st.header("Predictions")
        stock_ticker = st.text_input("Enter Stock Ticker", value="AAPL")
        model_type = st.selectbox("Select Model", [
            "LSTM", "XGBoost", "ARIMA", "Prophet", 
            "Random Forest", "Linear Regression", "Moving Average",
            "Holt-Winters", "ETS", "STL"
        ])
    
        # Add seasonality configuration for time series models
        if model_type in ["Holt-Winters", "ETS", "STL"]:
            seasonality_type = st.radio(
                "Select Seasonality Period",
                ["Weekly (5 days)", "Monthly (21 days)", "Quarterly (63 days)"],
                index=0  # Default to weekly
            )
            seasonal_periods = 5 if "Weekly" in seasonality_type else (21 if "Monthly" in seasonality_type else 63)
    
        if st.button("Generate Predictions"):
            stock_data = fetch_stock_data(stock_ticker)
            if not stock_data.empty:
                try:
                    predictions = None
                
                    # Machine Learning Models
                    if model_type == "LSTM":
                        if len(stock_data) < 60:
                            st.error("Error: Insufficient data for LSTM (requires at least 60 days).")
                        else:
                            model, scaler = train_lstm_model(stock_data)
                            predictions = predict_lstm(model, scaler, stock_data)

                    elif model_type == "XGBoost":
                        model = train_xgboost_model(stock_data)
                        predictions = predict_xgboost(model, stock_data)

                    elif model_type == "ARIMA":
                        model = train_arima_model(stock_data)
                        predictions = predict_arima(model)

                    elif model_type == "Prophet":
                        model = train_prophet_model(stock_data)
                        predictions = predict_prophet(model)

                    elif model_type == "Random Forest":
                        model = train_random_forest_model(stock_data)
                        predictions = predict_random_forest(model, stock_data)

                    elif model_type == "Linear Regression":
                        model = train_linear_regression_model(stock_data)
                        predictions = predict_linear_regression(model, stock_data)

                    elif model_type == "Moving Average":
                        predictions = predict_moving_average(stock_data)
                
                    # Time Series Models with configurable seasonality
                    elif model_type == "Holt-Winters":
                        model = ExponentialSmoothing(
                            stock_data['Close'],
                            trend='add',
                            seasonal='add',
                            seasonal_periods=seasonal_periods
                        ).fit()
                        predictions = model.forecast(30).values
                
                    elif model_type == "ETS":
                        model = ETSModel(
                            stock_data['Close'],
                            error='add',
                            trend='add',
                            seasonal='add',
                            seasonal_periods=seasonal_periods,
                            damped_trend=True
                        ).fit()
                        predictions = model.forecast(30)
                
                    elif model_type == "STL":
                        stl = STL(stock_data['Close'], period=seasonal_periods).fit()
                        last_trend = stl.trend.iloc[-1]
                        trend_slope = last_trend - stl.trend.iloc[-2]
                        future_trend = [last_trend + i*trend_slope for i in range(1, 31)]
                        seasonal_component = stl.seasonal.iloc[-seasonal_periods:].values[:30]
                        predictions = np.array(future_trend) + seasonal_component

                    # Visualization
                    if predictions is not None:
                        last_date = stock_data.index[-1]
                        future_dates = pd.date_range(start=last_date, periods=31, freq='B')[1:]
                    
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=stock_data.index,
                            y=stock_data['Close'],
                            mode='lines',
                            name='Historical Data'
                        ))
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=predictions,
                            mode='lines+markers',
                            name=f'Predicted ({model_type})',
                            line=dict(color='red')
                        ))
                    
                        # Add confidence intervals for probabilistic models
                        if model_type in ["ETS", "ARIMA", "Prophet"]:
                            if hasattr(model, 'get_prediction'):
                                pred_results = model.get_prediction(start=future_dates[0], end=future_dates[-1])
                                ci = pred_results.conf_int()
                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=ci.iloc[:, 0],
                                    fill=None,
                                    mode='lines',
                                    line=dict(width=0),
                                    showlegend=False
                                ))
                                fig.add_trace(go.Scatter(
                                    x=future_dates,
                                    y=ci.iloc[:, 1],
                                    fill='tonexty',
                                    mode='lines',
                                    line=dict(width=0),
                                    name='Confidence Interval'
                                ))
                    
                        fig.update_layout(
                            title=f"{stock_ticker} Price Forecast ({model_type}, {seasonality_type if model_type in ['Holt-Winters','ETS','STL'] else ''})",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig)
                    
                        # Model evaluation metrics
                        if len(stock_data) > 100:  # Only show if sufficient history
                            with st.expander("Model Performance Metrics"):
                                if model_type in ["Holt-Winters", "ETS", "STL"]:
                                    train = stock_data['Close'].iloc[:-30]
                                    test = stock_data['Close'].iloc[-30:]
                                    if model_type == "Holt-Winters":
                                        fit_model = ExponentialSmoothing(
                                            train,
                                            trend='add',
                                            seasonal='add',
                                            seasonal_periods=seasonal_periods
                                        ).fit()
                                    elif model_type == "ETS":
                                        fit_model = ETSModel(
                                            train,
                                            error='add',
                                            trend='add',
                                            seasonal='add',
                                            seasonal_periods=seasonal_periods
                                        ).fit()
                                
                                    preds = fit_model.forecast(30)
                                    mae = mean_absolute_error(test, preds)
                                    rmse = np.sqrt(mean_squared_error(test, preds))
                                    st.metric("MAE (30-day backtest)", f"${mae:.2f}")
                                    st.metric("RMSE (30-day backtest)", f"${rmse:.2f}")

                        predictions_chat(predictions, stock_ticker, model_type, stock_data)

                except Exception as e:
                    st.error(f"Error in {model_type} predictions: {str(e)}")
                    st.exception(e) if st.checkbox("Show technical details") else None

    if __name__ == "__main__":
            main()
