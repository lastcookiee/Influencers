import pandas as pd
import torch
import torch.nn as nn
import numpy as np

# Define the RL (DQN) model architecture.
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Configuration: must match your training setup.
state_dim = 20    # The campaign state is represented by 20 features.
action_dim = 1000 # RL model outputs Q-values for 1000 candidate influencers.
hidden_dim = 128

# Instantiate and load the RL model.
model = DQN(state_dim, action_dim, hidden_dim)
try:
    model.load_state_dict(torch.load("reinforcement_model.pth"))
    print("Loaded RL model weights.")
except Exception as e:
    print("Using random initialization for RL model:", e)
model.eval()

# Load influencer data from CSV.
# Use the first 1000 rows to match the action space.
df = pd.read_csv("influencer_data.csv")
candidates = df.head(action_dim).reset_index(drop=True)

# Simulate a campaign state (in practice, build this from real campaign data).
state = np.random.rand(state_dim).astype(np.float32)
state_tensor = torch.FloatTensor(state).unsqueeze(0)

# Get Q-values from the RL model for the current state.
with torch.no_grad():
    q_values = model(state_tensor).squeeze()  # shape: (1000,)

# Get the top 100 influencer indices based on Q-values.
topk = torch.topk(q_values, k=100)
top_indices = topk.indices.numpy()

# Retrieve the corresponding influencer information and sort by Q-value descending.
recommended_influencers = candidates.iloc[top_indices].copy()
recommended_influencers["q_value"] = q_values[top_indices].numpy()
recommended_influencers = recommended_influencers.sort_values(by="q_value", ascending=False)

print("Top 100 Recommended Influencers (RL):")
print(recommended_influencers[["influencer_id", "name", "gender", "category", "q_value"]])
