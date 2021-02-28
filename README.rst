=======
GulpIO2
=======

Binary storage format for deep learning on videos.

.. image:: https://github.com/willprice/GulpIO2/actions/workflows/test.yml/badge.svg
   :target: https://github.com/willprice/GulpIO2/actions

.. image:: https://readthedocs.org/projects/gulpio2/badge/?version=latest
   :target: https://gulpio2.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


Fork notice
===========

This is a fork of TwentyBN's `gulpio <https://github.com/TwentyBN/GulpIO>`_.
Key differences:

* **faster**: Uses ``simplejpeg`` for JPEG decoding by default (a wrapper around
  libjpeg-turbo) and replaces OpenCV with Pillow-SIMD for resizing images.
* **more flexible**: More frame access patterns are supported, not only ``slice``
  accesses.


Installation
============

.. code::

    pip install gulpio2


Migrating from gulpio
=====================

If you've been using `gulpio <https://github.com/TwentyBN/GulpIO>`_, then you can use ``gulpio2`` as a drop-in replacement. Simply replace all ``import gulpio ...`` statements with ``import gulpio2 ...``.


Prior Art
=========

* Inspired by: MXNet based RecordIO: http://mxnet.io/architecture/note_data_loading.html
* GulpIO: Original codebase https://github.com/TwentyBN/GulpIO


License
=======

All original gulpio code is Copyright (c) Twenty Billion Neurons and
licensed under the MIT License, see the file ``LICENSE.txt`` for details.
Subsequent code is Copyright (c) Will Price, and is also licensed under the MIT
License.
