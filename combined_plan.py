import json

from diet_vector import optimize_diet
from sleep_vector import optimize_sleep
from workout_vector import optimize_workout


def clamp(value, low=0.0, high=1.0):
    # Constrains value between low and high bounds using: clamp = max(low, min(value, high))
    return max(low, min(value, high))


def total_training_sets(workout):
    # Calculates total weekly training sets: chest + back + legs + shoulder + arm
    return (
        workout["chest_sets"]
        + workout["back_sets"]
        + workout["legs_sets"]
        + workout["shoulder_sets"]
        + workout["arm_sets"]
    )


def target_calories(user):
    # Calculates daily target calories based on goal: maintenance = weight * 30, then adds/subtracts based on goal
    # fat_loss: -400, hypertrophy: +400, strength: +250, default: maintenance
    weight = user.get("weight", 70)
    goal = user.get("goal", "hypertrophy")
    maintenance = weight * 30

    if goal == "fat_loss":
        return maintenance - 400
    if goal == "hypertrophy":
        return maintenance + 400
    if goal == "strength":
        return maintenance + 250
    return maintenance


def interaction_score(workout, diet, sleep, user):
    # Calculates compatibility score between workout, diet, and sleep by penalizing misalignments
    # Checks: protein >= max(weight*1.6, sets*1.4), calories >= target, sleep >= 7h + intensity bonus,
    # Returns: clamp(1 - total_penalty) scaled 0-1
    penalty = 0

    weekly_sets = total_training_sets(workout)
    required_protein = max(user.get("weight", 70) * 1.6, weekly_sets * 1.4)
    if diet["protein_g"] < required_protein:
        penalty += (required_protein - diet["protein_g"]) / required_protein

    expected_calories = target_calories(user)
    if diet["calories"] < expected_calories:
        penalty += 0.5 * ((expected_calories - diet["calories"]) / expected_calories)

    required_sleep = 7.0 + max(0, workout["intensity"] - 0.70) * 2
    if sleep["sleep_hours"] < required_sleep:
        penalty += (required_sleep - sleep["sleep_hours"]) / required_sleep

    weekly_minutes = workout["days_per_week"] * workout["session_minutes"]
    if weekly_minutes > 420 and diet["calories"] < expected_calories + 200:
        penalty += 0.15

    return round(clamp(1 - penalty), 4)


def build_combined_plan(user, seed=42):
    # Combines optimized workout, diet, and sleep vectors into a single plan
    # Computes weighted final score: 0.35*workout + 0.30*diet + 0.20*sleep + 0.15*compatibility
    workout_result = optimize_workout(user, seed=seed + 1)
    diet_result = optimize_diet(user, seed=seed + 2)
    sleep_result = optimize_sleep(user, seed=seed + 3)

    workout = workout_result["vector"]
    diet = diet_result["vector"]
    sleep = sleep_result["vector"]

    compatibility = interaction_score(workout, diet, sleep, user)
    final_score = (
        0.35 * workout_result["score"]
        + 0.30 * diet_result["score"]
        + 0.20 * sleep_result["score"]
        + 0.15 * compatibility
    )

    return {
        "user": user,
        "vectors": {
            "workout": workout,
            "diet": diet,
            "sleep": sleep,
        },
        "scores": {
            "workout": workout_result["score"],
            "diet": diet_result["score"],
            "sleep": sleep_result["score"],
            "interaction": compatibility,
            "final": round(final_score, 4),
        },
    }


if __name__ == "__main__":
    user_profile = {
        "weight": 70,
        "height": 170,
        "goal": "hypertrophy",
        "experience": "beginner",
    }

    result = build_combined_plan(user_profile)
    print(json.dumps(result, indent=2))
