This is the sample code for my article about fast feature extraction using Keras. You can find it at the following link.

[https://duongnt.com/fast-feature-extraction](https://duongnt.com/fast-feature-extraction)

## Usage

### Extract features from images

```
from image_to_hdf5 import write_data_to_hdf5

write_data_to_hdf5('dataset/train', 'hdf5_data/train.hdf5')
write_data_to_hdf5('dataset/valid', 'hdf5_data/valid.hdf5')
write_data_to_hdf5('dataset/test', 'hdf5_data/test.hdf5')
```

You can use the `HDF5Generator` class to yield data from `hdf5` files.
```
from hdf5_generator import HDF5Generator

with HDF5Generator('C:/Learn/Python/fast_feature_extraction/hdf5_data/test.hdf5', 32) as hdf5_gen:
    for images, labels in hdf5_gen.generator():
        print(images.shape, labels.shape)
        break
```

Will print
```
(32, 8192) (32,)
```

## Train a classifier on the extracted features

**Warning**: although training a single classifier does not take much time, we will perform hyperparameters tuning for 160 trials. On my low-end GPU, the whole process took around 90 minutes.

```
from fast_feature_extraction import start_hyper_tuning

start_hyper_tuning()
```

After tuning, the best model will be saved as `gender_prediction_ffe_best.keras`. Also, the test loss/test accuracy will also be printed to console (your loss/accuracy will be somewhat different because of the random initialization of the network weights).
```
[0.6207340359687805, 0.9470587968826294]
```

## License

MIT License

https://opensource.org/licenses/MIT