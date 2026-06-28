import random


SLEEP_BOUNDS = {
    "sleep_hours": (5.0, 10.0),
    "bedtime": (20.0, 27.0),
    "wake_time": (4.0, 10.0),
    "sleep_quality": (0.50, 1.00),
    "schedule_consistency": (0.50, 1.00),
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
    # Rounds all vector values to 2 decimal places
    return {key: round(value, 2) for key, value in position.items()}


def circular_hour_distance(actual, target):
    # Calculates shortest distance between hours on 24-hour cycle: min(diff, 24-diff)
    actual = actual % 24
    target = target % 24
    diff = abs(actual - target)
    return min(diff, 24 - diff)


def time_closeness(actual, target, tolerance=6):
    # Scores time closeness accounting for 24-hour cycle: clamp(1 - distance/tolerance)
    diff = circular_hour_distance(actual, target)
    return clamp(1 - diff / tolerance)


def target_sleep_hours(user):
    # Calculates target sleep duration: 7.5h base, +0.5h (strength), +0.25h (advanced)
    goal = user["goal"]
    experience = user["experience"]

    target = 7.5
    if goal == "strength":
        target = 8.0
    if experience == "advanced":
        target += 0.25

    return target


def target_bedtime(user):
    # Returns target bedtime: 22:30 (fat_loss), 23:00 (other goals)
    goal = user["goal"]
    if goal == "fat_loss":
        return 22.5
    return 23.0


def sleep_fitness(vector, user):
    # Weighted sleep quality score: 0.35*duration + 0.20*bedtime + 0.15*wake_time + 0.15*quality + 0.15*consistency
    duration_score = closeness(vector["sleep_hours"], target_sleep_hours(user))
    bedtime_score = time_closeness(vector["bedtime"], target_bedtime(user))

    expected_wake = (vector["bedtime"] + vector["sleep_hours"]) % 24
    wake_score = time_closeness(vector["wake_time"], expected_wake, tolerance=4)

    quality_score = clamp(vector["sleep_quality"])
    consistency_score = clamp(vector["schedule_consistency"])

    return clamp(
        0.35 * duration_score
        + 0.20 * bedtime_score
        + 0.15 * wake_score
        + 0.15 * quality_score
        + 0.15 * consistency_score
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
    # Particle Swarm Optimization: uses 40 particles with w=0.70, c1=c2=1.50
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


def optimize_sleep(user, seed=None):
    # Optimizes sleep plan using PSO to maximize sleep_fitness for user profile
    if seed is not None:
        random.seed(seed)
    return pso_maximize(
        fitness_fn=lambda vector: sleep_fitness(vector, user),
        bounds=SLEEP_BOUNDS,
    )
