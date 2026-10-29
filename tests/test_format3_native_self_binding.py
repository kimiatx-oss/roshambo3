"""Private format-3 library binder preserves native arithmetic and all bounds."""
import numpy as np
import pytest
from roshambo3.backend import cpu

def inputs(stride=3,zero_color=False):
    coords=np.zeros((2,6,stride),np.float32)
    coords[:,:,:3]=np.arange(36,dtype=np.float32).reshape(2,6,3)/8
    if stride==4:coords[:,:,3]=1.
    counts=np.tile(np.array([2,0,0] if zero_color else [2,1,1],np.uint8),(2,1))
    r=np.ones((3,3),np.float32);p=np.eye(3,dtype=np.float32)
    vaa=np.array([20.,30.],np.float64)
    colors=np.array([1.,np.nextafter(1.,2.)],np.float64)
    return coords,counts,r,p,vaa,colors

def prepare(private,data):
    coords,counts,r,p,vaa,colors=data
    if private:
        return cpu.extension()._prepare_r3g_library_self_overlaps(coords,counts,r,p,'native-binding-test',vaa,colors)
    return cpu.extension().prepare_combo_self_overlaps(np.empty((0,0,3),np.float32),
        np.empty((0,0),np.int32),np.empty(0,np.int32),coords,counts,r,p,
        'native-binding-test',None,vaa,colors,True,True)

@pytest.mark.parametrize('stride',[3,4])
@pytest.mark.parametrize('stored_color',[False,True])
@pytest.mark.parametrize('zero_color',[False,True])
def test_private_native_library_binding_is_exact(stride,stored_color,zero_color):
    data=list(inputs(stride,zero_color))
    if not stored_color:data[-1]=None
    expected=prepare(False,data);actual=prepare(True,data)
    np.testing.assert_array_equal(actual.shape_library_values.view(np.uint32),expected.shape_library_values.view(np.uint32))
    np.testing.assert_array_equal(actual.color_library_values.view(np.uint64),expected.color_library_values.view(np.uint64))
    assert actual.library_shape_from_stored_volume
    assert actual.library_color_from_sidecar==stored_color
    assert actual.shape_library_terms==expected.shape_library_terms==0
    assert actual.color_library_terms==expected.color_library_terms
    assert len(actual.shape_query_values)==0
    if stored_color:np.testing.assert_array_equal(actual.color_library_values.view(np.uint64),data[-1].view(np.uint64))
    elif zero_color:np.testing.assert_array_equal(actual.color_library_values,np.zeros(2,np.float64))

@pytest.mark.parametrize('fault',['coordinate_nan','negative_weight','counts_capacity','vaa_nan','vaa_negative','color_inf','vaa_rows','color_rows','count_width','count_rows','matrix_shape','matrix_nan','matrix_empty'])
def test_private_native_library_binding_refuses_malformed_inputs(fault):
    data=list(inputs(4))
    if fault=='coordinate_nan':data[0][0,0,0]=np.nan
    elif fault=='negative_weight':data[0][0,0,3]=-1
    elif fault=='counts_capacity':data[1][0,1]=6
    elif fault=='vaa_nan':data[4][0]=np.nan
    elif fault=='vaa_negative':data[4][0]=-1
    elif fault=='color_inf':data[5][0]=np.inf
    elif fault=='vaa_rows':data[4]=data[4][:1]
    elif fault=='color_rows':data[5]=data[5][:1]
    elif fault=='count_width':data[1]=data[1][:,:2].copy()
    elif fault=='count_rows':data[1]=data[1][:1]
    elif fault=='matrix_shape':data[2]=np.ones((2,3),np.float32)
    elif fault=='matrix_nan':data[2][1,1]=np.nan
    elif fault=='matrix_empty':data[2]=data[3]=np.empty((0,0),np.float32);data[1]=np.empty((2,0),np.uint8)
    with pytest.raises(ValueError):prepare(True,data)
