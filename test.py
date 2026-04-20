#Particle + user
class User:
    def __init__(self, weight, height, goal, experience):
        self.weight = weight
        self.height = height
        self.goal = goal
        self.experience = experience


class Particle:
    def __init__(self):
        self.workout = {
            "days": 4,
            "duration": 60,
            "chest": 12,
            "back": 14,
            "legs": 16,
            "shoulders": 10,
            "arms": 8,
            "intensity": 0.7,
            "reps": 10,
            "rest_gap": 2
        }

        self.diet = {
            "calories": 2500,
            "protein": 120,
            "carbs": 300,
            "fat": 70,
            "meal_freq": 3
        }

        self.sleep = {
            "sleep_hours": 7.5,
            "sleep_time": 23
        }

#helper functions
def clamp(x, min_val=0, max_val=1):
    return max(min_val, min(x, max_val))


def get_target_volume(user, muscle):
    base = {
        "chest": 14,
        "back": 16,
        "legs": 18,
        "shoulders": 12,
        "arms": 10
    }

    v = base[muscle]

    if user.goal == "fat_loss":
        v *= 0.8
    if user.goal == "strength":
        v *= 0.9
    if user.experience == "beginner":
        v *= 0.7

    return v


def get_muscle_weight(muscle):
    weights = {
        "chest": 0.25,
        "back": 0.25,
        "legs": 0.25,
        "shoulders": 0.15,
        "arms": 0.10
    }
    return weights[muscle]


def get_target_intensity(user):
    if user.goal == "fat_loss":
        return 0.65
    if user.goal == "hypertrophy":
        return 0.7
    if user.goal == "strength":
        return 0.85


def get_target_calories(user):
    maintenance = user.weight * 30
    if user.goal == "fat_loss":
        return maintenance - 400
    if user.goal == "hypertrophy":
        return maintenance + 400
    return maintenance


def get_target_protein(user):
    return user.weight * 1.8


def get_target_sleep(user):
    return 7.5

#workout function
def F_workout(Xw, user):
    volume_score = 0

    for muscle in ["chest", "back", "legs", "shoulders", "arms"]:
        target = get_target_volume(user, muscle)
        actual = Xw[muscle]

        s = 1 - abs(actual - target) / target
        s = clamp(s)

        volume_score += get_muscle_weight(muscle) * s

    intensity_target = get_target_intensity(user)
    intensity_score = clamp(
        1 - abs(Xw["intensity"] - intensity_target) / intensity_target
    )

    reps_score = clamp(1 - abs(Xw["reps"] - 10) / 10)

    total_time = Xw["days"] * Xw["duration"]
    time_penalty = max(0, (total_time - 360) / 360)

    return (
        0.4 * volume_score +
        0.3 * intensity_score +
        0.2 * reps_score -
        0.1 * time_penalty
    )

#diet function
def compute_macro_balance(carbs, protein, fat):
    total = carbs + protein + fat
    c = carbs / total
    p = protein / total
    f = fat / total

    deviation = abs(c - 0.5) + abs(p - 0.25) + abs(f - 0.25)
    return clamp(1 - deviation)


def F_diet(Xd, user):
    cal_target = get_target_calories(user)
    prot_target = get_target_protein(user)

    calorie_score = clamp(
        1 - abs(Xd["calories"] - cal_target) / cal_target
    )

    protein_score = clamp(
        Xd["protein"] / prot_target
    )

    macro_score = compute_macro_balance(
        Xd["carbs"], Xd["protein"], Xd["fat"]
    )

    return (
        0.4 * calorie_score +
        0.4 * protein_score +
        0.2 * macro_score
    )

#sleep function 
def compute_sleep_timing(time):
    ideal = 23
    diff = abs(time - ideal)

    if diff > 6:
        return 0
    return 1 - diff / 6


def F_sleep(Xs, user):
    target = get_target_sleep(user)

    duration_score = clamp(
        1 - abs(Xs["sleep_hours"] - target) / target
    )

    timing_score = compute_sleep_timing(Xs["sleep_time"])

    return 0.7 * duration_score + 0.3 * timing_score

#interaction function
def F_interaction(X, user):
    penalty = 0

    Xw = X.workout
    Xd = X.diet
    Xs = X.sleep

    total_volume = sum([
        Xw["chest"], Xw["back"], Xw["legs"],
        Xw["shoulders"], Xw["arms"]
    ])

    # Protein vs Volume
    required_protein = total_volume * 1.5
    if Xd["protein"] < required_protein:
        penalty += (required_protein - Xd["protein"]) / required_protein

    # Sleep vs Intensity
    required_sleep = get_target_sleep(user) + (Xw["intensity"] - 0.7) * 2
    if Xs["sleep_hours"] < required_sleep:
        penalty += (required_sleep - Xs["sleep_hours"]) / required_sleep

    # Calories vs Workload
    required_calories = get_target_calories(user)
    if Xd["calories"] < required_calories:
        penalty += (required_calories - Xd["calories"]) / required_calories

    return -penalty

#final fitness
def Fitness(X, user):
    Fw = F_workout(X.workout, user)
    Fd = F_diet(X.diet, user)
    Fs = F_sleep(X.sleep, user)
    Fi = F_interaction(X, user)

    return (
        0.4 * Fw +
        0.25 * Fd +
        0.2 * Fs +
        0.15 * Fi
    )

#TESTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
user = User(weight=70, height=170, goal="hypertrophy", experience="beginner")
particle = Particle()

score = Fitness(particle, user)
print("Fitness Score:", score)