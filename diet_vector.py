import random


DIET_BOUNDS = {
    "calories": (1200, 4500),
    "protein_g": (50, 260),
    "carbs_g": (50, 650),
    "fat_g": (25, 180),
    "meals_per_day": (2, 6),
    "water_liters": (1.5, 5.0),
}

INTEGER_KEYS = {
    "calories",
    "protein_g",
    "carbs_g",
    "fat_g",
    "meals_per_day",
}


def clamp(value, low=0.0, high=1.0):
    return max(low, min(value, high))


def closeness(actual, target):
    if target == 0:
        return 0
    return clamp(1 - abs(actual - target) / target)


def round_vector(position):
    vector = {}
    for key, value in position.items():
        if key in INTEGER_KEYS:
            vector[key] = int(round(value))
        else:
            vector[key] = round(value, 2)
    return vector


def target_calories(user):
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


def target_protein(user):
    weight = user.get("weight", 70)
    goal = user.get("goal", "hypertrophy")

    if goal == "fat_loss":
        return weight * 2.0
    if goal == "strength":
        return weight * 2.1
    return weight * 1.8


def target_water(user):
    weight = user.get("weight", 70)
    return weight * 0.035


def target_macro_ratio(user):
    goal = user.get("goal", "hypertrophy")
    if goal == "fat_loss":
        return {"carbs": 0.40, "protein": 0.35, "fat": 0.25}
    if goal == "strength":
        return {"carbs": 0.45, "protein": 0.30, "fat": 0.25}
    return {"carbs": 0.50, "protein": 0.25, "fat": 0.25}


def macro_calories(vector):
    return {
        "carbs": vector["carbs_g"] * 4,
        "protein": vector["protein_g"] * 4,
        "fat": vector["fat_g"] * 9,
    }


def macro_ratio_score(vector, user):
    macros = macro_calories(vector)
    total = sum(macros.values())
    if total <= 0:
        return 0

    target = target_macro_ratio(user)
    actual = {key: value / total for key, value in macros.items()}
    deviation = sum(abs(actual[key] - target[key]) for key in target)
    return clamp(1 - deviation)


def diet_fitness(vector, user):
    calories_score = closeness(vector["calories"], target_calories(user))
    protein_score = closeness(vector["protein_g"], target_protein(user))
    macro_score = macro_ratio_score(vector, user)

    macro_total = sum(macro_calories(vector).values())
    consistency_score = closeness(macro_total, vector["calories"])

    meal_score = closeness(vector["meals_per_day"], 4)
    water_score = closeness(vector["water_liters"], target_water(user))

    return clamp(
        0.30 * calories_score
        + 0.25 * protein_score
        + 0.20 * macro_score
        + 0.15 * consistency_score
        + 0.05 * meal_score
        + 0.05 * water_score
    )


def random_position(bounds):
    return {key: random.uniform(low, high) for key, (low, high) in bounds.items()}


def random_velocity(bounds):
    velocity = {}
    for key, (low, high) in bounds.items():
        span = high - low
        velocity[key] = random.uniform(-span, span) * 0.1
    return velocity


def pso_maximize(fitness_fn, bounds, swarm_size=40, iterations=150):
    swarm = []

    for _ in range(swarm_size):
        position = random_position(bounds)
        velocity = random_velocity(bounds)
        vector = round_vector(position)
        score = fitness_fn(vector)

        swarm.append(
            {
                "position": position,
                "velocity": velocity,
                "best_position": position.copy(),
                "best_score": score,
            }
        )

    best_particle = max(swarm, key=lambda particle: particle["best_score"])
    global_best_position = best_particle["best_position"].copy()
    global_best_score = best_particle["best_score"]
    history = [global_best_score]

    inertia = 0.72
    cognitive = 1.49
    social = 1.49

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

                next_value = current + particle["velocity"][key]
                particle["position"][key] = max(low, min(next_value, high))

            vector = round_vector(particle["position"])
            score = fitness_fn(vector)

            if score > particle["best_score"]:
                particle["best_score"] = score
                particle["best_position"] = particle["position"].copy()

                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle["position"].copy()

        history.append(global_best_score)

    return {
        "vector": round_vector(global_best_position),
        "score": round(global_best_score, 4),
        "history": history,
    }


def optimize_diet(user, seed=202):
    random.seed(seed)
    return pso_maximize(
        fitness_fn=lambda vector: diet_fitness(vector, user),
        bounds=DIET_BOUNDS,
    )
