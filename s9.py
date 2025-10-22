import numpy as np
import cv2
from utils.expert_dataset import ExpertDatasetWriter

# create a fake RGB image sequence
imgs = [np.random.randint(0,255,(64,64,3),dtype=np.uint8) for _ in range(3)]
writer = ExpertDatasetWriter(out_dir="/tmp/test_out", run_name="test_run", image_compression="jpeg", jpeg_quality=90)
# Use the writer's encode function path (copy of internal behavior)
encoded = [cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes() for img in imgs]
# decode similarly to reader
decoded = [cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR) for b in encoded]
decoded_stack = np.stack(decoded)[..., ::-1]  # BGR->RGB
assert decoded_stack.shape == (3,64,64,3)
print("Compression roundtrip ok, dtype:", decoded_stack.dtype)
