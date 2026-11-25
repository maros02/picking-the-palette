from fpsample import fps_sampling


def farthest_point_sample(X, sample_size, seed):
    return fps_sampling(X, sample_size, seed)