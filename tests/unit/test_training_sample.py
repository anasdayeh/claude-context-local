import numpy as np


def test_training_sample_capped(tmp_path):
    from search.training_sample import TrainingSampleStore

    store = TrainingSampleStore(tmp_path, max_vectors=10)
    for i in range(25):
        vec = np.ones(4, dtype=np.float32) * i
        store.add(vec, {"path": f"f{i}.py"})

    store.save()

    data = store.load()
    assert data["vectors"].shape[0] == 10
