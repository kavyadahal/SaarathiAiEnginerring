"""
Day 24 · Shared dataset — restaurant review corpus (same as Day 23)
Import it so every script uses the same data:
    from review_data import make_reviews
    texts, labels = make_reviews()
label 1 = positive review, 0 = negative review.
Needs:  nothing (pure Python)
"""

POSITIVE = [
    "the food was good and the service was fast",
    "good food and very friendly staff",
    "great service and great value for money",
    "the momo was delicious and the staff were friendly",
    "delicious food, fast service, good price",
    "excellent food and excellent service",
    "the curry was delicious and the staff was kind",
    "friendly staff and fresh food",
    "fresh ingredients and good flavour",
    "the dal bhat was delicious and hot",
    "good value and the service was quick",
    "quick service and delicious coffee",
    "the staff was friendly and the food was fresh",
    "great flavour and a clean place",
    "clean tables and very good food",
    "the thukpa was excellent and hot",
    "excellent value, the food was fresh",
    "we loved the food, service was fast",
    "the biryani was delicious and the portion was generous",
    "generous portion and good price",
    "the tea was good and the staff smiled",
    "friendly waiter and delicious pastries",
    "the food arrived hot and fresh",
    "hot food, fast service, friendly staff",
    "very good experience, we will come again",
    "the chef was great and the food was delicious",
    "great place, clean and friendly",
    "the salad was fresh and the service was good",
    "good coffee and a quick, friendly staff",
    "delicious flavour and excellent price",
]

NEGATIVE = [
    "the food was cold and the service was slow",
    "bad food and very rude staff",
    "terrible service and bad value for money",
    "the momo was cold and the staff were rude",
    "bad food, slow service, high price",
    "terrible food and terrible service",
    "the curry was bland and the staff was rude",
    "rude staff and stale food",
    "stale ingredients and no flavour",
    "the dal bhat was cold and bland",
    "bad value and the service was slow",
    "slow service and bitter coffee",
    "the staff was rude and the food was stale",
    "no flavour and a dirty place",
    "dirty tables and very bad food",
    "the thukpa was terrible and cold",
    "terrible value, the food was stale",
    "we hated the food, service was slow",
    "the biryani was dry and the portion was tiny",
    "tiny portion and high price",
    "the tea was bad and the staff ignored us",
    "rude waiter and stale pastries",
    "the food arrived cold and late",
    "cold food, slow service, rude staff",
    "very bad experience, we will not come again",
    "the chef was careless and the food was bland",
    "bad place, dirty and unfriendly",
    "the salad was stale and the service was bad",
    "bitter coffee and a slow, rude staff",
    "no flavour and terrible price",
]


def make_reviews():
    """Return (texts, labels). 1 = positive, 0 = negative."""
    texts = POSITIVE + NEGATIVE
    labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)
    return texts, labels


if __name__ == "__main__":
    texts, labels = make_reviews()
    print(f"{len(texts)} reviews | {sum(labels)} positive, {len(labels) - sum(labels)} negative")
