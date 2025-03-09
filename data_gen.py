import csv
import random

first_names = [
    "John", "Jane", "Alex", "Emily", "Michael", "Sarah", "David", "Olivia", "Daniel", "Sophia",
    "James", "Emma", "Robert", "Isabella", "William", "Mia", "Benjamin", "Charlotte", "Ethan", "Amelia",
    "Jacob", "Harper", "Noah", "Ava", "Lucas", "Ella", "Liam", "Grace", "Mason", "Chloe",
    "Logan", "Zoe", "Alexander", "Lily", "Henry", "Hannah", "Jackson", "Victoria", "Sebastian", "Avery",
    "Owen", "Scarlett", "Gabriel", "Aria", "Caleb", "Madison", "Nathan", "Evelyn"
]

last_names = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
    "Gomez"
]

categories = [
    "Fashion", "Tech", "Lifestyle", "Food", "Travel", "Fitness", "Beauty", "Gaming", "Music", "Sports",
    "Education", "Health", "Finance", "Entertainment", "DIY", "Automotive", "Home Decor", "Parenting", "Pets", "Art"
]

genders = ["Male", "Female", "Non-binary"]

filename = "influencer_data.csv"

with open(filename, "w", newline="") as csvfile:
    fieldnames = [
        "influencer_id", "name", "gender", "category", "followers_count",
        "engagement_rate", "trending_points", "avg_views",
        "avg_likes", "avg_comments", "past_campaign_success"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(1, 100001):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        gender = random.choice(genders)
        category = random.choice(categories)
        followers_count = random.randint(1000, 1000000)
        engagement_rate = round(random.uniform(0.01, 0.1), 3)  # e.g., 0.05 represents 5%
        trending_points = random.randint(100, 1000)
        avg_views = random.randint(500, 20000)
        avg_likes = random.randint(50, 5000)
        avg_comments = random.randint(10, 500)
        past_campaign_success = round(random.uniform(3.0, 5.0), 1)

        writer.writerow({
            "influencer_id": i,
            "name": name,
            "gender": gender,
            "category": category,
            "followers_count": followers_count,
            "engagement_rate": engagement_rate,
            "trending_points": trending_points,
            "avg_views": avg_views,
            "avg_likes": avg_likes,
            "avg_comments": avg_comments,
            "past_campaign_success": past_campaign_success
        })

print(f"CSV file '{filename}' with 100,000 rows has been generated.")
