from flask import Flask, jsonify, request
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

#####################################
# INFLUENCER RECOMMENDATION (RL MODEL)
#####################################

# Define the RL model architecture for influencer recommendation.
class InfluencerRecommender(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(InfluencerRecommender, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Configuration for the RL model.
state_dim = 20    # Number of features representing the campaign state.
action_dim = 1000 # Number of candidate influencers (must match CSV row count used for RL).
hidden_dim = 128

# Instantiate the RL model.
influencer_model = InfluencerRecommender(state_dim, action_dim, hidden_dim)
try:
    influencer_model.load_state_dict(torch.load("reinforcement_model.pth"))
    print("Loaded RL model for influencer recommendation.")
except Exception as e:
    print("RL model not found or error loading. Using random initialization.", e)
influencer_model.eval()

# Function to recommend influencers using the RL model.
def recommend_influencers():
    # Simulate a campaign state.
    state = np.random.rand(state_dim).astype(np.float32)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_values = influencer_model(state_tensor).squeeze()  # shape: (action_dim,)
    topk = torch.topk(q_values, k=100)
    indices = topk.indices.numpy()
    # Assume candidate influencers are stored in "influencer_data.csv"
    df_influencers = pd.read_csv("influencer_data.csv").head(action_dim).reset_index(drop=True)
    recommended = df_influencers.iloc[indices].copy()
    recommended['q_value'] = q_values[indices].numpy()
    recommended = recommended.sort_values(by='q_value', ascending=False)
    return recommended

#####################################
# BRAND RECOMMENDATION MODEL
#####################################

# Define the brand recommendation model with embeddings.
class BrandRecommendationModel(nn.Module):
    def __init__(self, num_influencers, num_brands, embedding_dim=32):
        super(BrandRecommendationModel, self).__init__()
        self.influencer_embedding = nn.Embedding(num_influencers, embedding_dim)
        self.brand_embedding = nn.Embedding(num_brands, embedding_dim)
        self.fc = nn.Linear(embedding_dim * 2, 1)
    
    def forward(self, influencer, brand):
        inf_emb = self.influencer_embedding(influencer)
        brand_emb = self.brand_embedding(brand)
        x = torch.cat([inf_emb, brand_emb], dim=1)
        return self.fc(x).squeeze()

# Load the brand recommendation dataset.
df_brand = pd.read_csv("influencer_brand_data.csv")

# If the target column is missing, create a synthetic one.
TARGET_COL = "collaboration_score"
if TARGET_COL not in df_brand.columns:
    df_brand["brand_budget"] = pd.to_numeric(df_brand["brand_budget"], errors='coerce')
    df_brand[TARGET_COL] = df_brand["brand_budget"].fillna(0) * df_brand["past_campaign_success"]

# Encode influencer_id and brand_id.
influencer_encoder = LabelEncoder()
brand_encoder = LabelEncoder()
df_brand['influencer_idx'] = influencer_encoder.fit_transform(df_brand['influencer_id'])
df_brand['brand_idx'] = brand_encoder.fit_transform(df_brand['brand_id'])
num_influencers_dataset = df_brand['influencer_idx'].nunique()
num_brands = df_brand['brand_idx'].nunique()

# Instantiate the brand recommendation model.
embedding_dim_brand = 32
brand_model = BrandRecommendationModel(num_influencers_dataset, num_brands, embedding_dim_brand)
try:
    brand_model.load_state_dict(torch.load("brand_recommendation_model.pth"))
    print("Loaded brand recommendation model weights.")
except Exception as e:
    print("Brand recommendation model not found or error loading.", e)
brand_model.eval()

# Function to recommend top K brands for a given influencer.
def recommend_brands_for_influencer(influencer_id, top_k=10):
    try:
        influencer_id = int(influencer_id)
    except ValueError:
        pass
    if influencer_id not in influencer_encoder.classes_:
        return None
    influencer_idx = influencer_encoder.transform([influencer_id])[0]
    influencer_tensor = torch.tensor([influencer_idx] * num_brands)
    brand_indices = torch.arange(num_brands)
    with torch.no_grad():
        scores = brand_model(influencer_tensor, brand_indices)
    topk = torch.topk(scores, k=top_k)
    top_brand_indices = topk.indices.numpy()
    unique_brands = df_brand[['brand_id', 'brand_name']].drop_duplicates().reset_index(drop=True)
    recommended_brands = unique_brands.iloc[top_brand_indices].copy()
    recommended_brands['predicted_score'] = scores[top_brand_indices].numpy()
    recommended_brands = recommended_brands.sort_values(by='predicted_score', ascending=False)
    return recommended_brands

#####################################
# API ROUTES
#####################################

# Previous influencer recommendation route.
@app.route('/rl_recommend', methods=['GET'])
def api_recommend_influencers():
    recommended = recommend_influencers()
    result = recommended.to_dict(orient='records')
    return jsonify(result)

# New brand recommendation route.
@app.route('/recommend_brands/<influencer_id>', methods=['GET'])
def api_recommend_brands(influencer_id):
    top_k = request.args.get('top_k', default=10, type=int)
    recommendations = recommend_brands_for_influencer(influencer_id, top_k=top_k)
    if recommendations is None:
        return jsonify({"error": "Influencer not found in dataset."}), 404
    result = recommendations.to_dict(orient='records')
    return jsonify(result)

#####################################
# RUN THE API SERVER
#####################################

if __name__ == '__main__':
    app.run(debug=True)
