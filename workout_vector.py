import random


WORKOUT_BOUNDS = {
    "days_per_week": (2, 6),
    "session_minutes": (45, 90),
    "chest_sets": (20, 24),
    "back_sets": (20, 26),
    "legs_sets": (20, 28),
    "shoulder_sets": (20, 22),
    "arm_sets": (20, 20),
    "intensity": (0.50, 0.95),
    "reps": (4, 15),
    "rest_gap": (1, 5),
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
    "rest_gap",
}

GOAL_SET_MULTIPLIERS = {
    "fat_loss": 0.85,
    "muscle_gain": 1.00,
    "strength": 0.90,
}

EXPERIENCE_SET_MULTIPLIERS = {
    "beginner": 0.70,
    "intermediate": 1.85,
    "advanced": 1.00,
}

GOAL_INTENSITY_TARGETS = {
    "fat_loss": 0.65,
    "muscle_gain": 0.72,
    "strength": 0.85,
}

GOAL_REP_TARGETS = {
    "fat_loss": 12,
    "muscle_gain": 10,
    "strength": 5,
}

GOAL_WEEKLY_MINUTES = {
    "fat_loss": 280,
    "muscle_gain": 220,
    "strength": 240,
}

EXPERIENCE_TIME_MULTIPLIERS = {
    "beginner": 0.75,
    "intermediate": 0.85,
    "advanced": 1.00,
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
    # Calculates target sets from workout bounds, adjusted by goal and experience
    goal = user["goal"]
    experience = user["experience"]

    low, high = WORKOUT_BOUNDS[muscle]
    value = (low + high) / 2
    return value * GOAL_SET_MULTIPLIERS[goal] * EXPERIENCE_SET_MULTIPLIERS[experience]

def target_intensity(user):
    # Returns target workout intensity: 0.85 (strength), 0.65 (fat_loss), 0.72 (muscle_gain)
    goal = user["goal"]
    return GOAL_INTENSITY_TARGETS[goal]


def target_reps(user):
    # Returns target reps per set: 5 (strength), 12 (fat_loss), 10 (muscle_gain)
    goal = user["goal"]
    return GOAL_REP_TARGETS[goal]


def target_weekly_minutes(user):
    # Calculates target weekly training minutes: base 220-280 adjusted by goal and experience
    goal = user["goal"]
    experience = user["experience"]

    return GOAL_WEEKLY_MINUTES[goal] * EXPERIENCE_TIME_MULTIPLIERS[experience]


def workout_fitness(vector, user):
    # Weighted workout quality score: volume, intensity, reps, weekly time, workload, and weekly rest-day consistency
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

    weekly_minutes = vector["days_per_week"] * vector["session_minutes"]
    time_score = closeness(weekly_minutes, target_weekly_minutes(user))

    total_sets = sum(vector[key] for key in muscle_keys)
    sets_per_day = total_sets / max(1, vector["days_per_week"])
    workload_score = clamp(1 - max(0, sets_per_day - 24) / 24)

    schedule_score = closeness(vector["days_per_week"] + vector["rest_gap"], 7)

    return clamp(
        0.35 * volume_score
        + 0.20 * intensity_score
        + 0.15 * reps_score
        + 0.10 * time_score
        + 0.10 * workload_score
        + 0.10 * schedule_score
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


def optimize_workout(user, seed=None):
    # Optimizes workout plan using PSO to maximize workout_fitness for user profile
    if seed is not None:
        random.seed(seed)
        
    return pso_maximize(
        fitness_fn=lambda vector: workout_fitness(vector, user),
        bounds=WORKOUT_BOUNDS,
    )
