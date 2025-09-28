import pandas as pd


def concatenate_restaurant_text(restaurant_df: pd.DataFrame) -> str:
    restaurant_details = []

    restaurant_details.append(f"Description: {restaurant_df['description']}")
    restaurant_details.append(f"Neighborhood: {restaurant_df['neighborhood']}")
    restaurant_details.append(f"Cuisines: {restaurant_df['cuisines']}")
    restaurant_details.append(f"Tags: {restaurant_df['tags']}")

    return " | ".join(restaurant_details)
