Usage
=====

'Gulp' a Dataset
----------------

The ``gulpio`` package has been designed to be infinitely hackable and to support
arbitrary datasets. To this end, we are providing a so-called *adapter
pattern*. Specifically there exists an abstract class in ``gulpio.adapters``:
the ``AbstractDatasetAdapter``.  In order to ingest your dataset, you basically
need to implement your own custom adapter that inherits from this.

You should be able to get going quickly by looking at the following examples,
that we use internally to gulp our video datasets.

* The class: ``gulpio.adapters.Custom20BNJsonAdapter`` `adapters.py <src/main/python/gulpio/adapters.py>`_
* The script: ``gulp2_20bn_json_videos`` `command line script <src/main/scripts/gulp2_20bn_json_videos>`_

And an example invocation would be:

.. code::

   gulp2_20bn_json_videos videos.json input_dir output_dir

Additionally, if you would like to ingest your dataset from the command line,
the ``register_adapter`` script can be used to generate the command line interface
for the new adapter. Write your adapter that inherits from the ``AbstractDatasetAdapter``
in the ``adapter.py`` file, then simply call:

.. code::

    gulp2_register_adapter gulpio.adapters <NewAdapterClassName>

The script that provides the command line interface will be in the main directory of the repository. To use it, execute ``./new_adapter_class_name``.


Sanity check the 'Gulped' Files
-------------------------------

A very basic test to check the correctness of the gulped files is provided by the ``gulp2_sanity_check`` script.
For execution run:

.. code::

    gulp2_sanity_check <folder containing the gulped files>

It tests:

* The presence of any content in the ``.gulp`` and ``.gmeta``-files
* The file size of the ``.gulp`` file corresponds to the required file size that is given in the ``.gmeta`` file
* Duplicate appearances of any video-ids

The file names of the files where any test fails will be printed. Currently no script to fix possible errors is
provided, 'regulping' is the only solution.


Read a 'Gulped' Dataset
-----------------------

In order to read from the gulps, you can let yourself be inspired by the
following snippet:

.. code:: python

    from gulpio2 import GulpDirectory
    # You can either read greyscale (`colorspace="GRAY"`) or RGB (`colorspace="RGB"`)
    # images.
    gulp_directory = GulpDirectory('/tmp/something_something_gulps')
    # iterate over all chunks
    for chunk in gulp_directory:
        # for each 'video' get the metadata and all frames
        for frames, meta in chunk:
            # do something with the metadata
            for i, f in enumerate(frames):
                # do something with the frames
                pass

Alternatively, a video with a specific ``id`` can be directly accessed via:

.. code:: python

    from gulpio2 import GulpDirectory
    gulp_directory = GulpDirectory('/tmp/something_something_gulps')
    frames, meta = gulp_directory[<id>]

For down-sampling or loading only a part of a video, a python slice or list of
indices can be passed as well:

.. code:: python

    frames, meta = gulp_directory[<id>, slice(1,10,2)]

    frames, meta = gulp_directory[<id>, [1, 5, 6, 8]]

or:

.. code:: python

    frames, meta = gulp_directory[<id>, 1:10:2]


Loading Data
------------

Below is an example loading an image dataset and defining an augmentation pipeline using torchvision.
Transformations are applied to each instance on the fly.

.. code:: python

    from torch.utils.data import DataLoader
    from torchvision.transforms import Scale, CenterCrop, Compose, Normalize


    class GulpImageDataset:
        def __init__(self, gulp_dir):
            self.gulp_dir = gulp_dir

        def __len__(self):
            return len(self.gulp_dir.merged_metadict)

    # define data augmentations. Notice that there are different functions for videos and images
    transforms = Compose([
        Resize(120),
        CenterCrop(112),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # define dataset wrapper and pick this up by the data loader interface.
    dataset = GulpDirectory('/path/to/train_data', transform=transforms)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)

    dataset_val = GulpImageDataset('/path/to/validation_data/', transform=transforms)
    loader_val = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=0, drop_last=True)

Here we iterate through the dataset we loaded. Iterator returns data and label as numpy arrays. You might need to cast these into the format of you
deep learning library.

.. code:: python

    for data, label in loader:
        # train your model here
        # ...
