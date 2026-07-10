from aquaticrs.dataset import Dataset
from aquaticrs.scene import Scene


def test_dataset_add_and_len() -> None:
    dataset = Dataset()
    dataset.add_scene(Scene(data="a"))

    assert len(dataset) == 1
