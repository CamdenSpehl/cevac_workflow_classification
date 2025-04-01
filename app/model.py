import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Define our own stopwords list instead of using NLTK
ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 
    'couldn', 'd', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 
    'from', 'further', 'had', 'hadn', 'has', 'hasn', 'have', 'haven', 'having', 'he', 'her', 'here', 'hers', 
    'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 'it', 'its', 'itself', 
    'just', 'll', 'm', 'ma', 'me', 'mightn', 'more', 'most', 'mustn', 'my', 'myself', 'needn', 'no', 'nor', 
    'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 
    'over', 'own', 're', 's', 'same', 'shan', 'she', 'should', 'shouldn', 'so', 'some', 'such', 't', 'than', 
    'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
    'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', 'we', 'were', 'weren', 'what', 
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'wouldn', 'y', 'you', 'your', 
    'yours', 'yourself', 'yourselves'
}

class WorkOrderClassifier:
    def __init__(self):
        """Initialize the WorkOrderClassifier model."""
        self.vectorizer = None
        self.model = None
        self.stop_words = ENGLISH_STOPWORDS
    
    def clean_text(self, text):
        """Clean the text by removing punctuation, converting to lowercase, and removing stop words."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = ' '.join([word for word in text.split() if word not in self.stop_words])  # Remove stop words
        return text
    
    def train(self, work_orders_df):
        """Train the classifier on the provided work orders dataframe."""
        # Prepare the data
        X = work_orders_df['WO_DESC'].apply(self.clean_text)
        y = work_orders_df['WO_SHOP']
        
        # Create and fit vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_vec = self.vectorizer.fit_transform(X)
        
        # Train the model
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_vec, y)
        
        return self
    
    def predict(self, work_order_desc):
        """Predict the shop for a given work order description."""
        if not self.model or not self.vectorizer:
            raise ValueError("Model has not been trained or loaded yet")
            
        # Clean and vectorize the input
        cleaned_desc = self.clean_text(work_order_desc)
        vec_desc = self.vectorizer.transform([cleaned_desc])
        
        # Make prediction
        shop_prediction = self.model.predict(vec_desc)[0]
        
        # Get probabilities if needed
        probas = self.model.predict_proba(vec_desc)[0]
        # Find index of the predicted class
        pred_idx = list(self.model.classes_).index(shop_prediction)
        confidence = probas[pred_idx]
        
        return {
            'shop': shop_prediction,
            'confidence': float(confidence)
        }
    
    def save(self, filepath):
        """Save the trained model and vectorizer to a file."""
        if not self.model or not self.vectorizer:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    def load(self, filepath):
        """Load a trained model and vectorizer from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        
        return self

# Function to train model from scratch
def train_model_from_data(data_path):
    """Train the model from scratch using data from the specified path."""
    # Load work order data
    df = pd.read_csv(data_path)
    
    # Filter to closed work orders with shop assignments
    closed_wos = df[df['WO_STATUS'] == 'CLOSED']
    closed_wos = closed_wos.dropna(subset=['WO_SHOP'])
    
    # Train the model
    classifier = WorkOrderClassifier()
    classifier.train(closed_wos)
    
    return classifier
