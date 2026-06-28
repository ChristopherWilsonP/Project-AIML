import random

# Diet vector (minimum, maximum)
DIET_BOUNDS = {
    "calories": (1200, 4500),
    "protein_g": (50, 260),
    "carbs_g": (80, 600),
    "fat_g": (25, 160),
    "meals_per_day": (2, 4),
    "water_liters": (2.0, 5.0),
}

INTEGER_KEYS = {
    "calories",
    "protein_g",
    "carbs_g",
    "fat_g",
    "meals_per_day",
}


def clamp(value, low=0.0, high=1.0):
    # Batasi nilai antara low dan high bounds
    return max(low, min(value, high))


def closeness(actual, target):
    # Hitung kedekatan: clamp(1 - |actual - target|/target)
    if target == 0:
        return 0
    return clamp(1 - abs(actual - target) / target)


def round_vector(position):
    # Bulatkan nilai vector: integer untuk key nutrient, 2 desimal untuk yang lain
    vector = {}
    for key, value in position.items():
        if key in INTEGER_KEYS:
            vector[key] = int(round(value))
        else:
            vector[key] = round(value, 2)
    return vector


def target_calories(user):
    # Hitung target kalori: maintenance = weight*30, disesuaikan berdasarkan goal
    # fat_loss: -400, muscle_gain: +400, strength: +250
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


def target_protein(user):
    # Hitung target protein harian: 1.5x untuk fat_loss, 1.8x untuk strength, 2.0x untuk muscle_gain
    weight = user["weight"]
    goal = user["goal"]

    if goal == "fat_loss":
        return weight * 1.5
    if goal == "strength":
        return weight * 1.8
    if goal == "muscle_gain":
        return weight * 2.0


def target_water(user):
    # Hitung target air harian: berat badan × 0.035 liter
    weight = user["weight"]
    return weight * 0.035


def target_macro_ratio(user):
    # Kembalikan rasio makro target berdasarkan goal
    goal = user["goal"]
    if goal == "fat_loss":
        return {"carbs": 0.40, "protein": 0.45, "fat": 0.15}
    if goal == "strength":
        return {"carbs": 0.45, "protein": 0.30, "fat": 0.25}
    if goal == "muscle_gain":
        return {"carbs": 0.40, "protein": 0.40, "fat": 0.20}

def macro_calories(vector):
    # Konversi gram makro ke kalori: carbs*4, protein*4, fat*9
    return {
        "carbs": vector["carbs_g"] * 4,
        "protein": vector["protein_g"] * 4,
        "fat": vector["fat_g"] * 9,
    }


def macro_ratio_score(vector, user):
    # Skor kecocokan rasio makro: clamp(1 - total deviasi)
    macros = macro_calories(vector)
    total = sum(macros.values())
    if total <= 0:
        return 0

    target = target_macro_ratio(user)
    actual = {key: value / total for key, value in macros.items()}
    deviation = sum(abs(actual[key] - target[key]) for key in target)
    return clamp(1 - deviation)


def diet_fitness(vector, user):
    # Skor kualitas diet terbobot: 0.30*calories + 0.25*protein + 0.20*macros + 0.15*consistency + 0.05*meals + 0.05*water
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
    # Buat posisi random dalam bounds untuk inisialisasi PSO
    return {key: random.uniform(low, high) for key, (low, high) in bounds.items()}


def random_velocity(bounds):
    # Buat velocity random: 10% rentang bounds dalam arah random
    velocity = {}
    for key, (low, high) in bounds.items():
        span = high - low
        velocity[key] = random.uniform(-span, span) * 0.1
    return velocity


def pso_maximize(fitness_fn, bounds, swarm_size=40, iterations=150):
    # Particle Swarm Optimization: pakai 40 particle dengan w=0.70, c1=c2=1.50
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

    w = 0.70
    c1 = 1.50
    c2 = 1.50

    for _ in range(iterations):
        for particle in swarm:
            for key, (low, high) in bounds.items():
                r1 = random.random()
                r2 = random.random()
                current = particle["position"][key]

                particle["velocity"][key] = (
                    w * particle["velocity"][key]
                    + c1 * r1 * (particle["best_position"][key] - current)
                    + c2 * r2 * (global_best_position[key] - current)
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


def optimize_diet(user, seed=None):
    # Optimasi rencana diet pakai PSO untuk maksimalkan diet_fitness sesuai profil user
    if seed is not None:
        random.seed(seed)
    return pso_maximize(
        fitness_fn=lambda vector: diet_fitness(vector, user),
        bounds=DIET_BOUNDS,
    )
