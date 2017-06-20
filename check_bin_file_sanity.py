import os

from pprint import pprint
from gulp_io import GulpVideoIO


def save_video():
    pass


if __name__ == "__main__":
    path_to_chunk = "/media/4TBSATA/kinetics/binary_data/chunk_gulp_1"
    path_bin = os.path.join(path_to_chunk, "train_data.bin")
    path_meta = os.path.join(path_to_chunk, "train_meta.pkl")

    vid_id = "3yaoNwz99xM"
    vid_path = os.path.join("sample_vids", vid_id)
    os.makedirs(vid_path, exist_ok=True)

    gulp_file = GulpVideoIO(path_bin, 'rb', path_meta)
    gulp_file.open()

    for i in range(len(gulp_file.meta_dict[vid_id])):
        img = gulp_file.read(gulp_file.meta_dict[vid_id][i])
        img.save(os.path.join(vid_path, "frame%04d.jpg" % i))
