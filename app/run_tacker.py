for frame_id, frame in enumerate(video):
    if frame_id % 20 == 0:
        detections = detect(frame)
    tracks = tracker.update(detections)
    visualize(tracks)
