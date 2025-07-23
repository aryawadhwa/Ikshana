# food_facts.py

# A simple dictionary to store food facts
food_facts_dict = {
    "apple": "Apples are rich in fiber, vitamins, and minerals. They are also low in calories and high in antioxidants.",
    "banana": "Bananas are high in potassium, which helps regulate blood pressure. They are also a good source of vitamin C.",
    "orange": "Oranges are an excellent source of vitamin C, which boosts the immune system and promotes skin health.",
    "carrot": "Carrots are a great source of beta-carotene, which is converted into vitamin A, essential for eye health.",
    "tomato": "Tomatoes are rich in lycopene, an antioxidant that has been linked to heart health and cancer prevention."
}

def food_facts(food_item):
    """
    Given a food item, print its nutritional facts.
    If the food item isn't in the dictionary, print a default message.
    """
    food_item = food_item.lower()  # Convert input to lowercase to make it case-insensitive
    if food_item in food_facts_dict:
        print(f"\t{food_facts_dict[food_item]}")
    else:
        print(f"\tSorry, no food facts available for {food_item}.")