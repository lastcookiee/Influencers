import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# Define the recommendation model architecture (must match the pretrained model)
class RecommendationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Set the input dimension to 20 as expected by the pretrained model
input_dim = 20   
hidden_dim = 128
output_dim = 1   # For example, a predicted recommendation score

# Instantiate the model
model = RecommendationModel(input_dim, hidden_dim, output_dim)

# Attempt to load the pretrained weights
try:
    model.load_state_dict(torch.load("pretrained_recommendation_model.pth"))
    print("Pretrained model loaded successfully.")
except Exception as e:
    print("Error loading pretrained model:", e)
model.eval()

# Load influencer data from CSV
df = pd.read_csv("influencer_data.csv")

# We originally have 7 numeric features:
feature_columns = [
    "followers_count", "engagement_rate", "trending_points",
    "avg_views", "avg_likes", "avg_comments", "past_campaign_success"
]
# Extract these features as a NumPy array (shape: [num_influencers, 7])
base_features = df[feature_columns].values.astype(np.float32)

# Since the model expects 20 features, pad the remaining 13 dimensions with zeros.
num_influencers = base_features.shape[0]
padding = np.zeros((num_influencers, input_dim - base_features.shape[1]), dtype=np.float32)
# Create a complete feature matrix of shape: [num_influencers, 20]
features = np.hstack([base_features, padding])

# Convert the features to a torch tensor
features_tensor = torch.FloatTensor(features)

# Use the model to predict a score for each influencer
with torch.no_grad():
    predictions = model(features_tensor).squeeze()  # shape: (num_influencers,)

# Add the predicted scores to the DataFrame and sort
df["predicted_score"] = predictions.numpy()
df_sorted = df.sort_values(by="predicted_score", ascending=False)

# Get the top 10 influencers as recommended by the model
top_influencers = df_sorted.head(10)

print("Top recommended influencers:")
print(top_influencers[["influencer_id", "name", "gender", "category", "predicted_score"]])
