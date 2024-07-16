def is_adjacent(scene: str, same_orbit: list):
    """
    This function makes a list of scene numbers from the same orbit as integers and checks
    if they are adjacent/sequential.
    """
    scene_nums = [int(scene.split(".")[-2].split("_")[-1]) for scene in same_orbit]
    return all(b - a == 1 for a, b in zip(scene_nums[:-1], scene_nums[1:]))
