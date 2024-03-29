from scipy.spatial.distance import cosine


def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)
