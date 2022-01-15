# -*- coding: utf-8 -*-

import h5py
import numpy as np


class HDF5Generator:
    def __init__(self, db_path, batch_size=32):
        self.batch_size = batch_size

        self.db = h5py.File(db_path)
        self.db_size = self.db['labels'].shape[0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def generator(self, max_epochs=np.inf):
        epochs = 0

        while epochs < max_epochs:
            for i in np.arange(0, self.db_size, self.batch_size):
                images = self.db['data'][i:i+self.batch_size]
                labels = self.db['labels'][i:i+self.batch_size]

                yield images, labels

    def close(self):
        self.db.close()
