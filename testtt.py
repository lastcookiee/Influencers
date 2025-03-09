import requests

def test_influencer_recommendation():
    url = "http://127.0.0.1:5000/rl_recommend"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json()
        print("Influencer Recommendation (RL):")
        for influencer in data:
            print(influencer)
    except requests.exceptions.RequestException as e:
        print("Error testing influencer recommendation endpoint:", e)

def test_brand_recommendation():
    # Replace with a valid influencer ID from your dataset.
    influencer_id = 1001  
    url = f"http://127.0.0.1:5000/recommend_brands/{influencer_id}?top_k=10"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad status codes
        data = response.json()
        print("Brand Recommendation for Influencer {}:".format(influencer_id))
        for brand in data:
            print(brand)
    except requests.exceptions.RequestException as e:
        print("Error testing brand recommendation endpoint:", e)

if __name__ == '__main__':
    print("Testing Influencer Recommendation Endpoint:")
    test_influencer_recommendation()
    print("\nTesting Brand Recommendation Endpoint:")
    test_brand_recommendation()
