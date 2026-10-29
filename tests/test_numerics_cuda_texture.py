"""Portable checks of diagnostic references; these do not qualify hardware."""
import numpy as np
from bench.numerics_cuda_texture import tensor_model


def test_tensor_reference_evaluates_multiaffine_polynomial():
    z,y,x=np.indices((3,3,3))
    nodes=np.stack((x*y*z,x*y+y*z+x*z,x+2*y-4*z,np.ones_like(x)),axis=-1)
    points=np.array([[.125,.25,.5],[1.5,.75,1.125]],np.float32)
    x,y,z=points.astype(np.float64).T
    expected=np.column_stack((x*y*z,x*y+y*z+x*z,x+2*y-4*z,np.ones(len(x))))
    np.testing.assert_array_equal(tensor_model(nodes,points),expected)


def test_tensor_reference_uses_border_zero_and_half_texel_addressing():
    nodes=np.ones((2,2,2,4),np.float16)
    points=np.array([[-.25,0,0],[-1,0,0],[1.25,0,0]],np.float32)
    np.testing.assert_array_equal(tensor_model(nodes,points),np.repeat([[.75],[0],[.75]],4,axis=1))
