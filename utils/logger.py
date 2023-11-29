import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging


def get_logger(log_dir, filename='train.log'):
    global logger
    logger = logging.getLogger(os.path.basename(log_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    h = logging.FileHandler(os.path.join(log_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def summarize(writer, global_step, scalars={}, vectors={}, histograms={}, images={},
              audios={}, audio_sr=44100):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in vectors.items():
        for i, d in enumerate(v):
            writer.add_scalar(k, d, i)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sr)
