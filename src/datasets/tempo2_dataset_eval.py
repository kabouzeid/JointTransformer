from src.datasets.tempo2_dataset import Tempo2Dataset


class Tempo2DatasetEval(Tempo2Dataset):
    def __getitem__(self, index):
        return self.getitem_tempo(index, eval=True)
