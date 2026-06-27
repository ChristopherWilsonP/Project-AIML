import random


WORKOUT_BOUNDS = {
    "days_per_week": (2, 6),
    "session_minutes": (30, 90),
    "chest_sets": (4, 24),
    "back_sets": (4, 26),
    "legs_sets": (4, 28),
    "shoulder_sets": (4, 22),
    "arm_sets": (4, 20),
    "intensity": (0.50, 0.95),
    "reps": (3, 15),
    "rest_minutes": (1.0, 4.0),
}

INTEGER_KEYS = {
    "days_per_week",
    "session_minutes",
    "chest_sets",
    "back_sets",
    "legs_sets",
    "shoulder_sets",
    "arm_sets",
    "reps",
}


def clamp(value, low=0.0, high=1.0):
    # Constrains value between low and high bounds
    return max(low, min(value, high))


def closeness(actual, target):
    # Calculates how close actual is to target: clamp(1 - |actual-target|/target)
    if target == 0:
        return 0
    return clamp(1 - abs(actual - target) / target)


def round_vector(position):
    # Rounds vector values: integers for exercise metrics, 2 decimals for others
    vector = {}
    for key, value in position.items():
        if key in INTEGER_KEYS:
            vector[key] = int(round(value))
        else:
            vector[key] = round(value, 2)
    return vector


def target_sets(user, muscle):
    # Calculates target sets for muscle group: base value adjusted by goal (0.85-1.15) and experience (0.70-1.15)
    goal = user.get("goal", "hypertrophy")
    experience = user.get("experience", "beginner")

    base = {
        "chest_sets": 14,
        "back_sets": 16,
        "legs_sets": 18,
        "shoulder_sets": 12,
        "arm_sets": 10,
    }

    value = base[muscle]
    if goal == "fat_loss":
        value *= 0.85
    elif goal == "strength":
        value *= 0.90

    if experience == "beginner":
        value *= 0.70
    elif experience == "advanced":
        value *= 1.15

    return value


def target_intensity(user):
    # Returns target workout intensity: 0.85 (strength), 0.65 (fat_loss), 0.72 (hypertrophy)
    goal = user.get("goal", "hypertrophy")
    if goal == "strength":
        return 0.85
    if goal == "fat_loss":
        return 0.65
    return 0.72


def target_reps(user):
    # Returns target reps per set: 5 (strength), 12 (fat_loss), 10 (hypertrophy)
    goal = user.get("goal", "hypertrophy")
    if goal == "strength":
        return 5
    if goal == "fat_loss":
        return 12
    return 10


def target_rest(user):
    # Returns target rest between sets: 3.0 min (strength), 1.5 min (fat_loss), 2.0 min (hypertrophy)
    goal = user.get("goal", "hypertrophy")
    if goal == "strength":
        return 3.0
    if goal == "fat_loss":
        return 1.5
    return 2.0


def target_weekly_minutes(user):
    # Calculates target weekly training minutes: base 220-280 adjusted by goal and experience
    goal = user.get("goal", "hypertrophy")
    experience = user.get("experience", "beginner")

    if goal == "fat_loss":
        target = 280
    elif goal == "strength":
        target = 240
    else:
        target = 220

    if experience == "beginner":
        target *= 0.85
    elif experience == "advanced":
        target *= 1.15

    return target


def workout_fitness(vector, user):
    # Weighted workout quality score: 0.35*volume + 0.20*intensity + 0.15*reps + 0.10*rest + 0.10*time + 0.10*workload
    muscle_keys = [
        "chest_sets",
        "back_sets",
        "legs_sets",
        "shoulder_sets",
        "arm_sets",
    ]
    muscle_weights = {
        "chest_sets": 0.23,
        "back_sets": 0.24,
        "legs_sets": 0.25,
        "shoulder_sets": 0.16,
        "arm_sets": 0.12,
    }

    volume_score = 0
    for key in muscle_keys:
        volume_score += muscle_weights[key] * closeness(vector[key], target_sets(user, key))

    intensity_score = closeness(vector["intensity"], target_intensity(user))
    reps_score = closeness(vector["reps"], target_reps(user))
    rest_score = closeness(vector["rest_minutes"], target_rest(user))

    weekly_minutes = vector["days_per_week"] * vector["session_minutes"]
    time_score = closeness(weekly_minutes, target_weekly_minutes(user))

    total_sets = sum(vector[key] for key in muscle_keys)
    sets_per_day = total_sets / max(1, vector["days_per_week"])
    workload_score = clamp(1 - max(0, sets_per_day - 24) / 24)

    return clamp(
        0.35 * volume_score
        + 0.20 * intensity_score
        + 0.15 * reps_score
        + 0.10 * rest_score
        + 0.10 * time_score
        + 0.10 * workload_score
    )


def random_position(bounds):
    # Generates random position vector within bounds for PSO initialization
    return {key: random.uniform(low, high) for key, (low, high) in bounds.items()}


def random_velocity(bounds):
    # Generates random velocity vector: 10% of bounds range in random direction
    velocity = {}
    for key, (low, high) in bounds.items():
        span = high - low
        velocity[key] = random.uniform(-span, span) * 0.1
    return velocity


def pso_maximize(fitness_fn, bounds, swarm_size=40, iterations=150):
    # Particle Swarm Optimization: uses 40 particles with inertia=0.72, cognitive=social=1.49
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


def optimize_workout(user, seed=101):
    # Optimizes workout plan using PSO to maximize workout_fitness for user profile
    random.seed(seed)
    return pso_maximize(
        fitness_fn=lambda vector: workout_fitness(vector, user),
        bounds=WORKOUT_BOUNDS,
    )
