import pickle
from collections import defaultdict

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re


def find_place_index(df, place):
    place = place.lower()

    if place in df['name'].values:
        return df[df['name'] == place].index[0]
    else:
        mask = (
                df['name'].str.lower().str.contains(place) |
                df['formatted_address'].str.lower().str.contains(place) |
                df['latest_reviews'].str.lower().str.contains(place)
        )
        indices = df[mask].index.tolist()
        if indices:
            return indices[0]
        else:
            return "No matches found"


def clean_name(name):
    return re.sub(r'[^a-zA-Z ]', '', name)


def destination_based_recommendation(destinations):
    places_df = pd.read_excel("data/Places Dataset.xlsx")
    places_response_df = pd.read_csv("data/places_response.csv")
    with open("states/similarity.pkl", "rb") as f:
        similarities = pickle.load(f)

    places_response_df['name'] = places_response_df['name'].apply(clean_name)

    recommendations = {}
    for destination in destinations:
        index = find_place_index(places_df, destination)
        if index != "No matches found":
            similarity_scores = list(enumerate(similarities[index]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            top_5_scores = similarity_scores[0:5]
            top_5_scores = [item for item in top_5_scores if item[1] != 0]

            recommendations[destination] = [places_df.iloc[i[0]]['name'] for i in top_5_scores]
        else:
            recommendations[destination] = "No matches found"

    recommended_places_details = {}
    for destination, places in recommendations.items():
        places_under_key = []
        for place in places:
            places = places_response_df[places_response_df['name'] == place.lower()]
            if not places.empty:
                places_under_key.append(places.values.tolist())
        recommended_places_details[destination] = places_under_key

    return recommended_places_details


def activity_based_recommendation(activities):
    with open("states/feature_vectors_activities.pkl", "rb") as f:
        feature_vectors_activities = pickle.load(f)

    with open("states/tfidf2.pkl", "rb") as f:
        tfidf = pickle.load(f)

    places_response_df = pd.read_csv("data/places_response.csv")
    places_df = pd.read_excel("data/Places Dataset.xlsx")

    places_response_df['name'] = places_response_df['name'].apply(clean_name)

    recommendations = {}

    for activity in activities:
        activity_vector = tfidf.transform([activity])
        similarity = cosine_similarity(feature_vectors_activities, activity_vector)
        similarity_scores = list(enumerate(similarity))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_5_scores = similarity_scores[0:5]
        top_5_scores = [item for item in top_5_scores if item[1] != 0]

        recommendations[activity] = [places_df.iloc[i[0]]['name'] for i in top_5_scores]

    recommended_places_details = {}

    for activity, places in recommendations.items():
        places_under_key = []
        for place in places:
            places = places_response_df[places_response_df['name'] == place.lower()]
            if not places.empty:
                places_under_key.append(places.values.tolist())
        recommended_places_details[activity] = places_under_key

    return recommended_places_details


def get_destination_and_activity_based_recommendations(destinations, activities):
    destination_based_recommendations = destination_based_recommendation(destinations)
    activity_based_recommendations = activity_based_recommendation(activities)

    recommendations = {
        "destination_based": destination_based_recommendations,
        "activity_based": activity_based_recommendations
    }

    return recommendations


def filter_by_rating(data, threshold=0.4):
    """
    Filters location data by rating. Only includes locations with a rating >= threshold.

    Parameters:
    - data: List of location entries (place name, lat, lon, rating).
    - threshold: The minimum rating required to include a location.

    Returns:
    - Filtered data with entries having rating >= threshold.
    """
    return [entry for entry in data if entry[0][3] >= threshold]


def clean_data(activity_locations, place_similarities, threshold=0.4):
    """
    Cleans both the activity locations and place similarities by removing entries with a rating below the threshold.

    Parameters:
    - activity_locations: Dictionary mapping activities to lists of location data.
    - place_similarities: Dictionary mapping places to lists of similar place data.
    - threshold: The minimum rating required to include a location (default is 0.4).

    Returns:
    - cleaned_activity_locations: Filtered activity locations.
    - cleaned_place_similarities: Filtered place similarities.
    """
    # Clean activity locations
    cleaned_activity_locations = {
        activity: filter_by_rating(locations, threshold)
        for activity, locations in activity_locations.items()
    }

    # Clean place similarities
    cleaned_place_similarities = {
        place: filter_by_rating(similar_places, threshold)
        for place, similar_places in place_similarities.items()
    }

    return cleaned_activity_locations, cleaned_place_similarities


def map_activities_to_places(activities, activity_locations, place_similarities):
    """
    Maps activities to bucket list places maximizing the number of activities and places covered.

    Parameters:
    - activities: List of activity names.
    - activity_locations: Dict mapping activities to lists of location data.
    - place_similarities: Dict mapping places to lists of similar place data.

    Returns:
    - place_activity_map: Dict mapping each place to a list of assigned activities.
    - total_activities: Number of activities covered.
    - total_places: Number of places visited.
    """

    # Initialize mappings
    place_activity_map = defaultdict(list)  # Mapping from place to list of activities
    covered_activities = set()
    covered_places = set()

    # Preprocess activity_locations: sort locations by similarity score descending for each activity
    sorted_activity_locations = {}
    for activity, locations in activity_locations.items():
        # Flatten the list of lists
        flat_locations = [loc[0] for loc in locations]
        # Sort by similarity score descending
        sorted_locs = sorted(flat_locations, key=lambda x: x[3], reverse=True)
        sorted_activity_locations[activity] = sorted_locs

    # Step 1: Assign activities to the most specific place from place_similarities
    for activity in activities:
        assigned = False
        for loc in sorted_activity_locations.get(activity, []):
            place_name = loc[0]  # Get the exact place name
            similarity_score = loc[3]
            for place, similar_places in place_similarities.items():
                for sim_place in similar_places:
                    if place_name == sim_place[0][0]:  # Match the exact place name
                        # Assign the activity to the more specific place name from `similar_places`
                        specific_place_name = sim_place[0][0]
                        place_activity_map[specific_place_name].append(activity)
                        covered_activities.add(activity)
                        covered_places.add(specific_place_name)
                        assigned = True
                        break
                if assigned:
                    break
            if assigned:
                break

    # Final counts
    total_activities = len(covered_activities)
    total_places = len(covered_places)

    return place_activity_map, total_activities, total_places


def suggest_additional_places(mapped_activities, cleaned_place_similarities, num_suggestions=5):
    """
    Suggests additional places if the total number of places in mapped_activities is less than num_suggestions.

    Parameters:
    - mapped_activities: The current map of places to activities.
    - cleaned_place_similarities: Filtered place similarities data.
    - num_suggestions: The number of total locations to suggest (default is 5).

    Returns:
    - final_suggestions: A list of final suggested sublocations, prioritized by uncovered categories.
    """
    # Start by checking how many unique places are already covered
    current_places = set(mapped_activities.keys())

    # If we already have 5 or more places, return the current places
    if len(current_places) >= num_suggestions:
        return list(current_places)

    # Track suggested places and sublocations
    suggested_places = set(current_places)  # Start with the current places
    covered_places = set(current_places)  # Track main places that are covered
    covered_sublocations = set()  # Track sublocations already suggested

    # Mark sublocations already covered by activities
    for place in mapped_activities:
        for similar_place in cleaned_place_similarities.get(place, []):
            covered_sublocations.add(similar_place[0][0])

    # List of places that have not yet been covered in the mapping
    uncovered_places = [place for place in cleaned_place_similarities if place not in mapped_activities]

    # Suggest from uncovered categories first
    for place in uncovered_places:
        if len(suggested_places) >= num_suggestions:
            break
        if place not in covered_places:  # Suggest from uncovered main places
            for sublocation in cleaned_place_similarities[place]:
                sublocation_name = sublocation[0][0]
                if sublocation_name not in covered_sublocations and place not in covered_places:
                    suggested_places.add(sublocation_name)  # Add the sublocation or place
                    covered_places.add(place)  # Mark the main place as covered
                    covered_sublocations.add(sublocation_name)  # Ensure no duplicate sublocations
                    break  # Suggest one sublocation per uncovered place

    # If we still need more suggestions after uncovered categories, suggest from already covered places
    if len(suggested_places) < num_suggestions:
        for place, sublocations in cleaned_place_similarities.items():
            if place in mapped_activities and place not in covered_places:
                for sublocation in sublocations:
                    sublocation_name = sublocation[0][0]
                    if sublocation_name not in covered_sublocations and place not in covered_places:
                        suggested_places.add(sublocation_name)  # Add the main place or sublocation
                        covered_places.add(place)  # Mark the main place as covered
                        covered_sublocations.add(sublocation_name)  # Ensure no duplicate sublocations
                        break  # Suggest only one sublocation from already covered places

            if len(suggested_places) >= num_suggestions:
                break

    return list(suggested_places)[:num_suggestions]  # Return only up to the number of suggestions


def fill_places_with_uncovered_activities(suggested_places, mapped_activities, activity_locations, num_suggestions=5):
    """
    Fills suggested_places with additional places from activity_locations if there are uncovered activities,
    until the total number of places reaches num_suggestions (5 by default).

    Parameters:
    - suggested_places: List of already suggested places.
    - mapped_activities: The current map of places to activities.
    - activity_locations: A dictionary mapping activities to lists of location data.
    - num_suggestions: The number of total places to suggest (default is 5).

    Returns:
    - final_suggestions: A list of suggested places, filled with uncovered activities if needed.
    """
    # Start by checking how many unique places we already have
    if len(suggested_places) >= num_suggestions:
        return suggested_places  # No need to add more if we already have enough

    # Identify activities that haven't been covered yet
    covered_activities = set(activity for activities in mapped_activities.values() for activity in activities)
    uncovered_activities = [activity for activity in activity_locations if activity not in covered_activities]

    # Suggest additional places for uncovered activities
    for activity in uncovered_activities:
        if len(suggested_places) >= num_suggestions:
            break  # Stop once we have enough suggestions
        # Find locations for the uncovered activity
        for location_data in activity_locations[activity]:
            place_name = location_data[0][0]
            if place_name not in suggested_places:
                suggested_places.append(place_name)  # Add the place
                break  # Only add one place per uncovered activity

    return suggested_places[:num_suggestions]  # Return up to num_suggestions


def fill_places_with_similarities(suggested_places, cleaned_place_similarities, num_suggestions=5):
    """
    Fills suggested_places with additional places from place_similarities if there are fewer than num_suggestions.

    Parameters:
    - suggested_places: List of already suggested places.
    - cleaned_place_similarities: Filtered place similarities data.
    - num_suggestions: The number of total places to suggest (default is 5).

    Returns:
    - final_suggestions: A list of suggested places, filled with uncovered activities if needed.
    """
    # Track suggested places to avoid duplicates
    existing_places = set(suggested_places)

    # Go through place_similarities to find additional places
    for place, sublocations in cleaned_place_similarities.items():
        if len(suggested_places) >= num_suggestions:
            break  # Stop once we reach the required number of suggestions
        for sublocation in sublocations:
            sublocation_name = sublocation[0][0]
            if sublocation_name not in existing_places:
                suggested_places.append(sublocation_name)  # Add the sublocation
                existing_places.add(sublocation_name)
                break  # Only add one sublocation per main place

    return suggested_places[:num_suggestions]  # Return up to num_suggestions


def get_best_places(activities, destinations):
    recommendations = get_destination_and_activity_based_recommendations(destinations, activities)

    activity_locations = recommendations["activity_based"]
    place_similarities = recommendations["destination_based"]

    cleaned_activity_locations, cleaned_place_similarities = clean_data(activity_locations, place_similarities)

    mapped_activities, activities_covered, places_visited = map_activities_to_places(
        activities,
        cleaned_activity_locations,
        cleaned_place_similarities
    )

    suggested_places = suggest_additional_places(mapped_activities, cleaned_place_similarities)

    suggested_places = fill_places_with_uncovered_activities(suggested_places, mapped_activities, activity_locations)

    suggested_places = fill_places_with_similarities(suggested_places, place_similarities)

    return suggested_places

