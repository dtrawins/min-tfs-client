from min_tfs_client.requests import TensorServingClient
from min_tfs_client.tensors import tensor_proto_to_ndarray
import numpy as np
import datetime

imgs = np.load('imgs.npy', mmap_mode='r', allow_pickle=False)
#imgs = imgs.transpose((0,2,3,1))
imgs = imgs - np.min(imgs)  # Normalization 0-255
imgs = imgs / np.ptp(imgs) * 255  # Normalization 0-255
img = imgs[0:1]

print(img.shape)

client = TensorServingClient(host="127.0.0.1", port=9000, credentials=None)
start_time = datetime.datetime.now()
response = client.predict_request(
    model_name="resnet",
    model_version=1,
    tensor_content=True,
    input_dict={
        # These input keys are model-specific
        "data": img
    },
)
end_time = datetime.datetime.now()
duration = (end_time - start_time).total_seconds() * 1000
print("duration", duration, "ms")
print(response)
float_output = tensor_proto_to_ndarray(response.outputs["prob"])
#print(float_output)

print(float_output.shape)
print(np.argmax(float_output))
