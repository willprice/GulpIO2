# v0.0.4

Bugfix: `gulpio2.utils.burst_video_into_frames` extracted frames with the naming
format `%04d.jpg`, so if more than 1000 frames were extracted they would not be
gulped correctly. This has now been changed to `%010d.jpg` which will handle 60
FPS video a duration of 316 years correctly!
