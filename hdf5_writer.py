# -*- coding: utf-8 -*-

import h5py


class HDF5Writer:
    def __init__(self, output_path, buffer_size, dims):
        self.output_path = output_path
        self.buffer_size = buffer_size
        self.dims = dims

        self.db = h5py.File(output_path, 'w')
        self.data = self.db.create_dataset('data', dims, dtype='float32')
        self.labels = self.db.create_dataset('labels', dims[0], dtype = 'int')

        self.buffer = {
            'data': [],
            'labels': []
        }

        self.idx = 0 # Index in database

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def write(self, data, label):
        self.buffer['data'].append(data)
        self.buffer['labels'].append(label)

        # The buffer is full, write it to disk
        if (len(self.buffer['data']) >= self.buffer_size):
            self.flush()

    def flush(self):
        # Write buffer to disk
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']

        self.idx = i

        # Reset buffer
        self.buffer = {
            'data': [],
            'labels': []
        }
    
    def close(self):
        # If buffer still contains data, flush it all to disk
        if len(self.buffer['data']) > 0:
            self.flush()

        # Close database
        self.db.close()
