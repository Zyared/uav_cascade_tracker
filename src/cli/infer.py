import cv2
from omegaconf import OmegaConf
from pathlib import Path
from ..pipelines.cascade_pipeline import CascadePipeline
from ..io.video_reader import VideoReader
from ..io.video_writer import VideoWriter


def main(cfg_path: str = "configs/default.yaml"):
    cfg = OmegaConf.load(cfg_path)
    pipe = CascadePipeline(cfg)


    vr = VideoReader(cfg.video.source)
    writer = None


    for i, frame in enumerate(vr):
        out = pipe.step(frame)
        if cfg.video.display:
            cv2.imshow("cascade", out)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    if cfg.video.write_output:
        if writer is None:
            h, w = out.shape[:2]
            writer = VideoWriter(cfg.video.out_path, fps=25, size=(w, h))
        writer.write(out)


    if writer:
        writer.close()
    cv2.destroyAllWindows()
    

    if __name__ == "__main__":
        main()