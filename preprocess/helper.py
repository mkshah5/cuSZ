import numpy as np

t = np.fromfile('./tensor_dims-26_dtype-complex128.bin')

treal = t.real
tim = t.imag

treal_32 = np.float32(treal)
tim_32 = np.float32(tim)

treal_32.tofile('./real_tensor_d26_f32.bin')
tim_32.tofile('./im_tensor_d26_f32.bin')