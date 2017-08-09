import os
import tempfile
import sh
import json
import glob
import atexit
import shutil

import numpy as np
import numpy.testing as npt

from gulpio.fileio import GulpChunk

# step 0: setup

temp_dir = tempfile.mkdtemp(prefix='gulpio-integration-test-')
print(temp_dir)
atexit.register(shutil.rmtree, temp_dir)
output_dir = os.path.join(temp_dir, 'output')

black_frame = np.zeros((100, 100, 3))
white_frame = np.ones((100, 100, 3)) * 255


def check_generated_files():
    gulps = glob.glob(os.path.join(output_dir, '*.gulp'))
    for gulp in sorted(gulps):
        chunk_id = gulp.split('_')[-1].split('.')[0]
        gulp_chunk = GulpChunk(chunk_id, output_dir,
                               expected_chunks=3)
        with gulp_chunk.open('rb') as fp:
            for frames, m in gulp_chunk.read_chunk(fp):
                for i, f in enumerate(frames):
                    arr = np.array(f)
                    if i % 2 != 0:
                        npt.assert_array_equal(white_frame, arr)
                    else:
                        npt.assert_array_equal(black_frame, arr)


# step 1: generate the videos and JSON file

with open(os.path.join(temp_dir, 'source'), 'wb') as source_frames:
    for _ in range(60):
        source_frames.write(b'\x00' * 30000)
        source_frames.write(b'\xff' * 30000)

sh.ffmpeg('-t', '10',
          '-s', '100x100',
          '-f', 'rawvideo',
          '-pix_fmt', 'rgb24',
          '-r', '12',
          '-i', './source',
          'source.mp4',
          _cwd=temp_dir,
          )

json_content = []

for i in range(25):
    vid_id = str(i).zfill(3)
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
expected_sizes = [988800, 988800, 296640, 335, 22311, 22321, 6599]

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
    '--extend',
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
expected_sizes = [988800, 988800, 296640, 197760, 28, 22311, 22321, 6599, 4351]

assert expected_sizes == sizes

# step 8: ungulp the videos (from the extended files) and check the result

check_generated_files()
