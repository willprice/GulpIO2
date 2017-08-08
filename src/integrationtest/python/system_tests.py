import os
import sys
import tempfile
import sh
import json

# step 1: generate the videos

temp_dir = tempfile.mkdtemp(prefix='gulpio-integration-test-')
print(temp_dir)
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

for i in range(103):
    vid_id = str(i).zfill(3)
    os.makedirs(os.path.join(temp_dir, vid_id))
    sh.cp('source.mp4',
          os.path.join(vid_id, 'source_' + vid_id + '.mp4'),
          _cwd=temp_dir)
    json_content.append({'id': vid_id, 'template': 'ANY_' + vid_id})

with open(os.path.join(temp_dir, 'videos.json'), 'w') as fp:
    json.dump(json_content, fp)



# step 2: run the gulping

#PATH=src/main/scripts:$PATH PYTHONPATH=src/main/python
sh.gulp_20bn_json_videos(
    '--videos_per_chunk',  '10',
    '--extend',
    os.path.join(temp_dir, 'videos.json'),
    temp_dir,
    os.path.join(temp_dir, 'output'),
)

# step 3: sanity check the output

# step 4: ungulp the videos

# step 5: check the result

sys.exit(1)
