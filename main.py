from utils import read_video, write_video
from trackers import Tracker

def main():

    # Read video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #Initialize the tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub = True,
                                       stub_path = 'stubs/track_stubs.pkl')

    write_video(video_frames, 'output_videos/output_video.mp4')

if __name__ == "__main__":
    main()