from pathlib import Path
from absl import app, flags
import os
import numpy as np
import time
import tqdm

from score.both import get_inception_and_fid_score

FLAGS = flags.FLAGS

flags.DEFINE_string("eval_output_dir", default = None, help="Evaluation direcoty of the output images")
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')


def load_image_dir_to_tensor(image_path):

    files = [os.path.join(image_path, f) for f in os.listdir(image_path)]
    images = []
    for f in tqdm.tqdm(files, desc="loading imgs"):
        img = np.load(f)
        images.append(img) # [C,H,W]
    images = np.stack(images, axis=0)
    print(images.shape)
    return images

def evaluate(image_path):
    images = load_image_dir_to_tensor(image_path)
    print("calculating FID and IS scores")
    start = time.time()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=None,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    print(f"Finished, time used {time.time() - start}")
    return (IS, IS_std), FID, images
    

def eval_from_output_dir():
    (IS, IS_std), FID, _ = evaluate(FLAGS.eval_output_dir)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    with open(str(Path(FLAGS.eval_output_dir).parent / "eval_result.txt"), "w") as f:
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID), file = f)

def main(argv):
    eval_from_output_dir()


if __name__ == "__main__":

    app.run(main)
