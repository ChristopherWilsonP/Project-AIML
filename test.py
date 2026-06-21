from dataclasses import dataclass
import json
import random


@dataclass
class User:
    weight: float
    height: float
    goal: str
    experience: str


@dataclass
class Plan:
    workout: dict
    diet: dict
    sleep: dict


def clamp(x, min_val=0, max_val=1):
    return max(min_val, min(x, max_val))


def clamp_to_bounds(value, bounds):
    low, high = bounds
    return max(low, min(value, high))


def get_target_volume(user, muscle):
    base = {
        "chest": 14,
        "back": 16,
        "legs": 18,
        "shoulders": 12,
        "arms": 10,
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
        "arms": 0.10,
    }
    return weights[muscle]


def get_target_intensity(user):
    if user.goal == "fat_loss":
        return 0.65
    if user.goal == "hypertrophy":
        return 0.7
    if user.goal == "strength":
        return 0.85
    return 0.7


def get_target_reps(user):
    if user.goal == "strength":
        return 5
    if user.goal == "fat_loss":
        return 12
    return 10


def get_target_rest_gap(user):
    if user.goal == "strength":
        return 3
    if user.goal == "fat_loss":
        return 1.5
    return 2


def get_target_calories(user):
    maintenance = user.weight * 30
    if user.goal == "fat_loss":
        return maintenance - 400
    if user.goal == "hypertrophy":
        return maintenance + 400
    return maintenance


def get_target_protein(user):
    if user.goal == "strength":
        return user.weight * 2.0
    return user.weight * 1.8


def get_target_sleep(user):
    if user.goal == "strength":
        return 8
    return 7.5


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

    reps_target = get_target_reps(user)
    reps_score = clamp(1 - abs(Xw["reps"] - reps_target) / reps_target)

    rest_target = get_target_rest_gap(user)
    rest_score = clamp(1 - abs(Xw["rest_gap"] - rest_target) / rest_target)

    total_time = Xw["days"] * Xw["duration"]
    time_penalty = max(0, (total_time - 360) / 360)

    return (
        0.4 * volume_score
        + 0.25 * intensity_score
        + 0.2 * reps_score
        + 0.15 * rest_score
        - 0.1 * time_penalty
    )


def compute_macro_balance(carbs, protein, fat):
    carb_calories = carbs * 4
    protein_calories = protein * 4
    fat_calories = fat * 9
    total = carb_calories + protein_calories + fat_calories
    if total <= 0:
        return 0

    c = carb_calories / total
    p = protein_calories / total
    f = fat_calories / total

    deviation = abs(c - 0.5) + abs(p - 0.25) + abs(f - 0.25)
    return clamp(1 - deviation)


def F_diet(Xd, user):
    cal_target = get_target_calories(user)
    prot_target = get_target_protein(user)

    calorie_score = clamp(1 - abs(Xd["calories"] - cal_target) / cal_target)
    protein_score = clamp(1 - abs(Xd["protein"] - prot_target) / prot_target)
    macro_score = compute_macro_balance(Xd["carbs"], Xd["protein"], Xd["fat"])

    meal_freq_score = clamp(1 - abs(Xd["meal_freq"] - 4) / 4)

    return (
        0.35 * calorie_score
        + 0.35 * protein_score
        + 0.2 * macro_score
        + 0.1 * meal_freq_score
    )


def compute_sleep_timing(time):
    ideal = 23
    diff = min(abs(time - ideal), 24 - abs(time - ideal))

    if diff > 6:
        return 0
    return 1 - diff / 6


def F_sleep(Xs, user):
    target = get_target_sleep(user)

    duration_score = clamp(1 - abs(Xs["sleep_hours"] - target) / target)
    timing_score = compute_sleep_timing(Xs["sleep_time"])

    return 0.7 * duration_score + 0.3 * timing_score


def F_interaction(plan, user):
    penalty = 0

    Xw = plan.workout
    Xd = plan.diet
    Xs = plan.sleep

    total_volume = sum(
        [Xw["chest"], Xw["back"], Xw["legs"], Xw["shoulders"], Xw["arms"]]
    )

    required_protein = total_volume * 1.5
    if Xd["protein"] < required_protein:
        penalty += (required_protein - Xd["protein"]) / required_protein

    required_sleep = get_target_sleep(user) + max(0, Xw["intensity"] - 0.7) * 2
    if Xs["sleep_hours"] < required_sleep:
        penalty += (required_sleep - Xs["sleep_hours"]) / required_sleep

    required_calories = get_target_calories(user)
    if Xd["calories"] < required_calories:
        penalty += (required_calories - Xd["calories"]) / required_calories

    return clamp(1 - penalty)


def Fitness(plan, user):
    Fw = F_workout(plan.workout, user)
    Fd = F_diet(plan.diet, user)
    Fs = F_sleep(plan.sleep, user)
    Fi = F_interaction(plan, user)

    return 0.4 * Fw + 0.25 * Fd + 0.2 * Fs + 0.15 * Fi


def round_position(position, integer_keys):
    rounded = {}
    for key, value in position.items():
        if key in integer_keys:
            rounded[key] = int(round(value))
        else:
            rounded[key] = round(value, 2)
    return rounded


def create_random_position(bounds):
    return {
        key: random.uniform(low, high)
        for key, (low, high) in bounds.items()
    }


def create_random_velocity(bounds):
    velocity = {}
    for key, (low, high) in bounds.items():
        span = high - low
        velocity[key] = random.uniform(-span, span) * 0.1
    return velocity


def pso_optimize(
    name,
    fitness_function,
    bounds,
    integer_keys=None,
    swarm_size=35,
    iterations=120,
    inertia=0.72,
    cognitive=1.49,
    social=1.49,
):
    integer_keys = set(integer_keys or [])

    swarm = []
    for _ in range(swarm_size):
        position = create_random_position(bounds)
        velocity = create_random_velocity(bounds)
        evaluated_position = round_position(position, integer_keys)
        score = fitness_function(evaluated_position)

        swarm.append(
            {
                "position": position,
                "velocity": velocity,
                "best_position": position.copy(),
                "best_score": score,
            }
        )

    global_best_particle = max(swarm, key=lambda particle: particle["best_score"])
    global_best_position = global_best_particle["best_position"].copy()
    global_best_score = global_best_particle["best_score"]
    history = [global_best_score]

    for _ in range(iterations):
        for particle in swarm:
            for key, (low, high) in bounds.items():
                r1 = random.random()
                r2 = random.random()
                current = particle["position"][key]

                particle["velocity"][key] = (
                    inertia * particle["velocity"][key]
                    + cognitive * r1 * (particle["best_position"][key] - current)
                    + social * r2 * (global_best_position[key] - current)
                )

                next_position = current + particle["velocity"][key]
                particle["position"][key] = clamp_to_bounds(next_position, (low, high))

            evaluated_position = round_position(particle["position"], integer_keys)
            score = fitness_function(evaluated_position)

            if score > particle["best_score"]:
                particle["best_score"] = score
                particle["best_position"] = particle["position"].copy()

                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle["position"].copy()

        history.append(global_best_score)

    best_vector = round_position(global_best_position, integer_keys)
    return {
        "name": name,
        "best_vector": best_vector,
        "best_score": global_best_score,
        "history": history,
    }


def optimize_plan(user, seed=42):
    random.seed(seed)

    workout_bounds = {
        "days": (2, 6),
        "duration": (30, 90),
        "chest": (4, 22),
        "back": (4, 24),
        "legs": (4, 26),
        "shoulders": (4, 20),
        "arms": (4, 18),
        "intensity": (0.5, 0.95),
        "reps": (3, 15),
        "rest_gap": (1, 4),
    }

    diet_bounds = {
        "calories": (1200, 4500),
        "protein": (50, 250),
        "carbs": (50, 600),
        "fat": (30, 160),
        "meal_freq": (2, 6),
    }

    sleep_bounds = {
        "sleep_hours": (5, 10),
        "sleep_time": (18, 29),
    }

    workout_result = pso_optimize(
        "workout",
        lambda x: F_workout(x, user),
        workout_bounds,
        integer_keys={
            "days",
            "duration",
            "chest",
            "back",
            "legs",
            "shoulders",
            "arms",
            "reps",
        },
    )

    diet_result = pso_optimize(
        "diet",
        lambda x: F_diet(x, user),
        diet_bounds,
        integer_keys={"calories", "protein", "carbs", "fat", "meal_freq"},
    )

    sleep_result = pso_optimize(
        "sleep",
        lambda x: F_sleep(x, user),
        sleep_bounds,
    )

    plan = Plan(
        workout=workout_result["best_vector"],
        diet=diet_result["best_vector"],
        sleep=sleep_result["best_vector"],
    )

    return {
        "plan": plan,
        "workout_result": workout_result,
        "diet_result": diet_result,
        "sleep_result": sleep_result,
        "final_score": Fitness(plan, user),
        "interaction_score": F_interaction(plan, user),
    }


def print_section(title, data):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in data.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    user = User(weight=70, height=170, goal="hypertrophy", experience="beginner")
    result = optimize_plan(user)
    plan = result["plan"]

    print("PSO Optimization Result")
    print("=======================")
    print(f"User: {user}")

    print_section(
        f"Best Workout Vector (score={result['workout_result']['best_score']:.4f})",
        plan.workout,
    )
    print_section(
        f"Best Diet Vector (score={result['diet_result']['best_score']:.4f})",
        plan.diet,
    )
    print_section(
        f"Best Sleep Vector (score={result['sleep_result']['best_score']:.4f})",
        plan.sleep,
    )

    merged_plan = {
        "workout": plan.workout,
        "diet": plan.diet,
        "sleep": plan.sleep,
    }

    print("\nMerged Plan Vector")
    print("------------------")
    print(json.dumps(merged_plan, indent=2))

    print("\nCombined Score")
    print("--------------")
    print(f"Workout score: {F_workout(plan.workout, user):.4f}")
    print(f"Diet score: {F_diet(plan.diet, user):.4f}")
    print(f"Sleep score: {F_sleep(plan.sleep, user):.4f}")
    print(f"Interaction score: {result['interaction_score']:.4f}")
    print(f"Final fitness score: {result['final_score']:.4f}")
