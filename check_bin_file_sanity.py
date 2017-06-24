import os
import sys

from pprint import pprint
from gulpio import GulpVideoIO


def save_video():
    pass


if __name__ == "__main__":
    path_to_chunk = "/media/4TBSATA/kinetics/binary_data/from_videos/validation_ffmpeg/"
    path_bin = os.path.join(path_to_chunk, "data0.bin")
    path_meta = os.path.join(path_to_chunk, "meta0.bin")

    vid_id = "6IbvOJxXnOo"
    vid_path = os.path.join("sample_vids", vid_id)
    os.makedirs(vid_path, exist_ok=True)

    gulp_file = GulpVideoIO(path_bin, 'rb', path_meta)
    gulp_file.open()
    pprint(gulp_file.meta_dict['6IbvOJxXnOo'])
    for i in range(len(gulp_file.meta_dict[vid_id])):
        img = gulp_file.read(gulp_file.meta_dict[vid_id][i])
        img.save(os.path.join(vid_path, "frame%04d.jpg" % (i + 1)))
