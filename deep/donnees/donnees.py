from typing import Iterator, NamedTuple

from deep.tenseurs.tenseur import Tenseur, np

Batch = NamedTuple("Batch", [("inputs", Tenseur), ("target", Tenseur)])


class DonneesIterateur:
    def __call__(self, inputs: Tenseur, target: Tenseur) -> Iterator[Batch]:
        pass


class BatchIterateur(DonneesIterateur):
    def __init__(self, batch_size: int = 32, shuffle: bool = True):

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tenseur, target: Tenseur) -> Iterator[Batch]:

        starts = np.arange(0, len(inputs), self.batch_size)

        if self.shuffle == True:
            np.random.shuffle(starts)

        for start in starts:

            end = start + self.batch_size
            input_batch = inputs[start:end]
            target_batch = target[start:end]
            yield Batch(input_batch, target_batch)
