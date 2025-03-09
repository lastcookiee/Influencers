import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

# Load the dataset
df = pd.read_csv("influencer_brand_data.csv")
print("Columns in dataset:", df.columns.tolist())

# Specify the target column name; if not present, create one synthetically.
TARGET_COL = "collaboration_score"
if TARGET_COL not in df.columns:
    print(f"Target column '{TARGET_COL}' not found. Creating synthetic target.")
    # Ensure brand_budget is numeric
    df["brand_budget"] = pd.to_numeric(df["brand_budget"], errors='coerce')
    # Create a synthetic target: collaboration_score = brand_budget * past_campaign_success.
    df[TARGET_COL] = df["brand_budget"].fillna(0) * df["past_campaign_success"]

# Encode influencer_id and brand_id into numeric indices
influencer_encoder = LabelEncoder()
brand_encoder = LabelEncoder()
df['influencer_idx'] = influencer_encoder.fit_transform(df['influencer_id'])
df['brand_idx'] = brand_encoder.fit_transform(df['brand_id'])

num_influencers = df['influencer_idx'].nunique()
num_brands = df['brand_idx'].nunique()
print(f"Number of influencers: {num_influencers}, Number of brands: {num_brands}")

# Define a custom dataset
class InfluencerBrandDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        influencer = row['influencer_idx']
        brand = row['brand_idx']
        score = row[TARGET_COL]  # target value
        return influencer, brand, score

dataset = InfluencerBrandDataset(df)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the brand recommendation model with embeddings
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
        out = self.fc(x)
        return out.squeeze()

embedding_dim = 32
model = BrandRecommendationModel(num_influencers, num_brands, embedding_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model for a few epochs (for demonstration)
num_epochs = 5
for epoch in range(num_epochs):
    epoch_loss = 0
    for influencer, brand, score in dataloader:
        optimizer.zero_grad()
        predictions = model(influencer, brand)
        loss = criterion(predictions, score.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

# Save the trained model's state dictionary
torch.save(model.state_dict(), "brand_recommendation_model.pth")
print("Model saved to 'brand_recommendation_model.pth'")
