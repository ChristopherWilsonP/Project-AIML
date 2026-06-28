import json

from diet_vector import optimize_diet
from sleep_vector import optimize_sleep
from workout_vector import optimize_workout


VALID_GOALS = {"fat_loss", "muscle_gain", "strength"}
VALID_EXPERIENCES = {"beginner", "intermediate", "advanced"}
REQUIRED_USER_FIELDS = {"weight", "height", "goal", "experience"}
WEEK_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def clamp(value, low=0.0, high=1.0):
    # Constraint nilai antara low dan high bounds menggunakan: clamp = max(low, min(value, high))
    return max(low, min(value, high))


def total_training_sets(workout):
    # Hitung total set mingguan: chest + back + legs + shoulder + arm
    return (
        workout["chest_sets"]
        + workout["back_sets"]
        + workout["legs_sets"]
        + workout["shoulder_sets"]
        + workout["arm_sets"]
    )


def build_training_schedule(days_per_week):
    templates = {
        2: ["Monday", "Thursday"],
        3: ["Monday", "Wednesday", "Friday"],
        4: ["Monday", "Tuesday", "Thursday", "Saturday"],
        5: ["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"],
        6: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    }

    training_days = templates.get(days_per_week, WEEK_DAYS[:days_per_week])
    rest_days = [day for day in WEEK_DAYS if day not in training_days]

    return {
        "training_days": training_days,
        "rest_days": rest_days,
    }


def calculate_bmi(user):
    height_m = user["height"] / 100
    return user["weight"] / (height_m ** 2)


def bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "normal"
    if bmi < 30:
        return "overweight"
    if bmi < 40:
        return "obese"
    return "severely_obese"


def bmi_warnings(bmi):
    if bmi < 18.5:
        return [
            "BMI rendah. Program sebaiknya fokus pada peningkatan berat badan yang bertahap dan aman."
        ]
    if bmi >= 40:
        return [
            "BMI sangat tinggi. Output tetap berupa optimasi program, tetapi user ini perlu diperlakukan hati-hati."
        ]
    if bmi >= 30:
        return [
            "BMI tinggi. Program tetap bisa digunakan, tetapi target latihan dan diet sebaiknya dibuat bertahap."
        ]
    return []


def bmi_score_factor(bmi):
    if bmi < 16:
        return 0.55
    if bmi < 18.5:
        return 0.75
    if bmi < 25:
        return 1.00
    if bmi < 30:
        return 0.90
    if bmi < 40:
        return 0.75
    return 0.60


def profile_assessment(user):
    bmi = calculate_bmi(user)
    return {
        "bmi": round(bmi, 2),
        "bmi_category": bmi_category(bmi),
        "bmi_score_factor": bmi_score_factor(bmi),
        "warnings": bmi_warnings(bmi),
    }


def target_calories(user):
    # Hitung target kalori harian berdasarkan goal: maintenance = weight * 30, disesuaikan dengan goal
    # fat_loss: -400, muscle_gain: +400, strength: +250, default: maintenance
    weight = user["weight"]
    goal = user["goal"]
    maintenance = weight * 30

    if goal == "fat_loss":
        return maintenance - 400
    if goal == "muscle_gain":
        return maintenance + 400
    if goal == "strength":
        return maintenance + 250
    return maintenance


def interaction_score(workout, diet, sleep, user):
    # Hitung skor kompatibilitas workout, diet, dan sleep dengan penalizing ketidaksesuaian
    # Cek: protein >= max(weight*1.6, sets*1.4), calori >= target, sleep >= 7h + intensity bonus,
    # Kembalikan: clamp(1 - total_penalty) dalam skala 0-1
    penalty = 0

    weekly_sets = total_training_sets(workout)
    required_protein = max(user["weight"] * 1.6, weekly_sets * 1.4)
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

    if workout["days_per_week"] + workout["rest_gap"] != 7:
        penalty += 0.15

    return round(clamp(1 - penalty), 4)


def build_combined_plan(user, seed=None):
    # Gabungkan vector workout, diet, dan sleep yang sudah dioptimasi jadi satu plan
    # Hitung skor akhir terbobot: 0.35*workout + 0.30*diet + 0.20*sleep + 0.15*compatibility
    user = validate_user(user)

    if seed is None:
        workout_result = optimize_workout(user)
        diet_result = optimize_diet(user)
        sleep_result = optimize_sleep(user)
    else:
        workout_result = optimize_workout(user, seed=seed + 1)
        diet_result = optimize_diet(user, seed=seed + 2)
        sleep_result = optimize_sleep(user, seed=seed + 3)

    workout = workout_result["vector"]
    diet = diet_result["vector"]
    sleep = sleep_result["vector"]
    training_schedule = build_training_schedule(workout["days_per_week"])
    assessment = profile_assessment(user)

    compatibility = interaction_score(workout, diet, sleep, user)
    optimization_final = (
        0.35 * workout_result["score"]
        + 0.30 * diet_result["score"]
        + 0.20 * sleep_result["score"]
        + 0.15 * compatibility
    )
    final_score = optimization_final * assessment["bmi_score_factor"]

    return {
        "user": user,
        "profile_assessment": assessment,
        "vectors": {
            "workout": workout,
            "diet": diet,
            "sleep": sleep,
        },
        "schedule": training_schedule,
        "scores": {
            "workout": workout_result["score"],
            "diet": diet_result["score"],
            "sleep": sleep_result["score"],
            "interaction": compatibility,
            "optimization_final": round(optimization_final, 4),
            "final": round(final_score, 4),
        },
    }


def validate_user(user):
    missing = REQUIRED_USER_FIELDS - set(user)
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Input user belum lengkap: {fields}")

    if user["goal"] not in VALID_GOALS:
        options = ", ".join(sorted(VALID_GOALS))
        raise ValueError(f"Goal tidak valid. Pilih salah satu: {options}")

    if user["experience"] not in VALID_EXPERIENCES:
        options = ", ".join(sorted(VALID_EXPERIENCES))
        raise ValueError(f"Experience tidak valid. Pilih salah satu: {options}")

    if user["weight"] <= 0:
        raise ValueError("Weight harus lebih dari 0")

    if user["height"] <= 0:
        raise ValueError("Height harus lebih dari 0")

    return user


def ask_float(prompt):
    while True:
        value = input(prompt).strip()
        try:
            number = float(value)
        except ValueError:
            print("Masukkan angka yang valid.")
            continue

        if number <= 0:
            print("Nilai harus lebih dari 0.")
            continue

        return number


def ask_choice(prompt, choices):
    choices_text = ", ".join(sorted(choices))
    while True:
        value = input(f"{prompt} ({choices_text}): ").strip().lower()
        if value in choices:
            return value
        print(f"Pilihan tidak valid. Gunakan salah satu: {choices_text}")


def get_user_input():
    return {
        "weight": ask_float("Masukkan berat badan (kg): "),
        "height": ask_float("Masukkan tinggi badan (cm): "),
        "goal": ask_choice("Masukkan goal", VALID_GOALS),
        "experience": ask_choice("Masukkan experience", VALID_EXPERIENCES),
    }


if __name__ == "__main__":
    user_profile = get_user_input()
    result = build_combined_plan(user_profile)
    print(json.dumps(result, indent=2))
