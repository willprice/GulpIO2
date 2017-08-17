import os
import tempfile
import sh
import json
import atexit
import shutil

import numpy as np
import numpy.testing as npt

from gulpio import GulpDirectory

# step 0: setup

temp_dir = tempfile.mkdtemp(prefix='gulpio-integration-test-')
print(temp_dir)
atexit.register(shutil.rmtree, temp_dir)
output_dir = os.path.join(temp_dir, 'output')

black_frame = np.zeros((100, 100, 3))
white_frame = np.ones((100, 100, 3)) * 255


def check_frames(frames, color=None):
    for i, f in enumerate(frames):
        arr = np.array(f)
        if color == 'white':
            npt.assert_array_equal(white_frame, arr)
        elif color == 'black':
            npt.assert_array_equal(black_frame, arr)
        elif color == 'alternating':
            if i % 2 != 0:
                npt.assert_array_equal(white_frame, arr)
            else:
                npt.assert_array_equal(black_frame, arr)


def check_generated_files():
    # check all frames
    gulp_directory = GulpDirectory(output_dir)
    for chunk in gulp_directory.chunks():
        with chunk.open('rb'):
            for frames, meta in chunk.read_all():
                check_frames(frames)

    # check random access for a few videos
    frames, meta = gulp_directory[0]
    check_frames(frames, 'alternating')
    frames, meta = gulp_directory[(11, slice(0, None, 2))]
    check_frames(frames, 'black')
    frames, meta = gulp_directory[(21, slice(1, None, 2))]
    check_frames(frames, 'white')


# step 1: generate the videos and JSON file

with open(os.path.join(temp_dir, 'source'), 'wb') as source_frames:
    for _ in range(40):
        source_frames.write(b'\x00' * 30000)
        source_frames.write(b'\xff' * 30000)

sh.ffmpeg('-t', '10',
          '-s', '100x100',
          '-f', 'rawvideo',
          '-pix_fmt', 'rgb24',
          '-r', '8',
          '-i', './source',
          'source.mp4',
          _cwd=temp_dir,
          )

json_content = []

for i in range(25):
    vid_id = str(i)
    os.makedirs(os.path.join(temp_dir, vid_id))
    sh.cp('source.mp4',
          os.path.join(vid_id, 'source_' + vid_id + '.mp4'),
          _cwd=temp_dir)
    json_content.append({'id': vid_id, 'template': 'ANY_' + vid_id})

with open(os.path.join(temp_dir, 'videos.json'), 'w') as fp:
    json.dump(json_content[:23], fp)


# step 2: run the gulping

# PATH=src/main/scripts:$PATH PYTHONPATH=src/main/python
command = sh.gulp_20bn_json_videos(
    '--videos_per_chunk',  '10',
    os.path.join(temp_dir, 'videos.json'),
    temp_dir,
    output_dir,
)

# step 3: sanity check the output

files = sorted(os.listdir(output_dir))
expected_files = [
    'data_0.gulp',
    'data_1.gulp',
    'data_2.gulp',
    'label2idx.json',
    'meta_0.gmeta',
    'meta_1.gmeta',
    'meta_2.gmeta'
]
assert expected_files == files

sizes = [os.path.getsize(os.path.join(output_dir, f)) for f in files]
expected_sizes = [659200, 659200, 197760, 302, 15059, 15083, 4430]
print(expected_sizes, sizes)
assert expected_sizes == sizes

# step 4: ungulp the videos and check the result

check_generated_files()

# step 5: write a second JSON files for extending

with open(os.path.join(temp_dir, 'videos_extend.json'), 'w') as fp:
    json.dump(json_content[23:], fp)

# step 6: extend the existing gulps

command = sh.gulp_20bn_json_videos(
    '--videos_per_chunk',  '10',
    os.path.join(temp_dir, 'videos_extend.json'),
    temp_dir,
    output_dir,
)

# step 7: sanity check the extended output

files = sorted(os.listdir(output_dir))
expected_files = [
    'data_0.gulp',
    'data_1.gulp',
    'data_2.gulp',
    'data_3.gulp',
    'label2idx.json',
    'meta_0.gmeta',
    'meta_1.gmeta',
    'meta_2.gmeta',
    'meta_3.gmeta',
]
assert expected_files == files

sizes = [os.path.getsize(os.path.join(output_dir, f)) for f in files]
expected_sizes = [659200, 659200, 197760, 131840, 26, 15059, 15083, 4430, 2905]
print(expected_sizes, sizes)
assert expected_sizes == sizes

# step 8: ungulp the videos (from the extended files) and check the result

check_generated_files()
