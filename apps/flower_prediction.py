import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="wide"
)

@st.cache_data
def load_model_data():
    """Load and prepare the Iris dataset."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return X, y, iris.target_names

def train_model(X, y):
    """Train a Random Forest model."""
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def create_feature_plot(X, y, target_names, feature_x, feature_y):
    """Create a scatter plot of two features."""
    df = X.copy()
    df['Species'] = [target_names[t] for t in y]
    fig = px.scatter(
        df,
        x=feature_x,
        y=feature_y,
        color='Species',
        title=f'{feature_x} vs {feature_y}',
        template='simple_white'
    )
    return fig

def main():
    st.title('🌸 Iris Flower Species Predictor')
    
    st.markdown("""
    ## About This App
    This application uses machine learning to predict the species of Iris flowers based on their measurements.
    Simply adjust the sliders below to input flower measurements and get a prediction!
    
    The model is trained on the famous Iris dataset and uses a Random Forest Classifier.
    """)
    
    # Load data and train model
    X, y, target_names = load_model_data()
    model = train_model(X, y)
    
    # Create two columns for input and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Flower Measurements")
        # Input sliders
        sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
        sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
        petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.7, 0.1)
        petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.4, 0.1)
        
        # Make prediction
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display prediction
        st.header("Prediction")
        predicted_species = target_names[prediction[0]]
        st.success(f"Predicted Species: **{predicted_species}**")
        
        # Show prediction probabilities
        st.subheader("Prediction Confidence")
        proba_df = pd.DataFrame({
            'Species': target_names,
            'Probability': prediction_proba
        })
        st.dataframe(proba_df, use_container_width=True)
    
    with col2:
        st.header("Data Visualization")
        
        # Feature selection for plot
        st.subheader("Explore Feature Relationships")
        feature_x = st.selectbox('Select X-axis feature:', X.columns)
        feature_y = st.selectbox('Select Y-axis feature:', X.columns, index=1)
        
        # Create and display plot
        fig = create_feature_plot(X, y, target_names, feature_x, feature_y)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mark user's input on the plot
        st.markdown("### Your Input")
        input_df = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value': [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.dataframe(input_df, use_container_width=True)

    # Add feature importance section
    st.header("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_importance = px.bar(
        importance_df,
        x='Feature',
        y='Importance',
        title='Feature Importance in Prediction',
        template='simple_white'
    )
    st.plotly_chart(fig_importance, use_container_width=True)

if __name__ == "__main__":
    main()
