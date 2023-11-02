import torch
import numpy as np

def coco2h36m(coco_input):
    '''
        Input: coco_input (N x T x J x C)
               N: number of people
               T: number of frames
               J: number of bodyjoints (i.e., 17 body joints)
               C: number of dimensions
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho
               6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip
               12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    
    is_torch, is_image = False, False
    if isinstance(coco_input, torch.Tensor):
        coco_input = coco_input.numpy()
        is_torch = True
    
    if coco_input.ndim == 3:
        N, J, C = coco_input.shape
        coco_input.reshape(N, 1, J, C) # Assume the image as a single frame
        is_image = True
    
    N, T, J, C = coco_input.shape

    h36m_output = np.zeros((N, T, J, C))
    
    # Mapping from COCO to H36M joints
    h36m_output[:,:,0,:]  = (coco_input[:,:,11,:] + coco_input[:,:,12,:]) * 0.5
    h36m_output[:,:,1,:]  = coco_input[:,:,12,:]
    h36m_output[:,:,2,:]  = coco_input[:,:,14,:]
    h36m_output[:,:,3,:]  = coco_input[:,:,16,:]
    h36m_output[:,:,4,:]  = coco_input[:,:,11,:]
    h36m_output[:,:,5,:]  = coco_input[:,:,13,:]
    h36m_output[:,:,6,:]  = coco_input[:,:,15,:]
    h36m_output[:,:,8,:]  = (coco_input[:,:,5,:] + coco_input[:,:,6,:]) * 0.5
    h36m_output[:,:,7,:]  = (h36m_output[:,:,0,:] + h36m_output[:,:,8,:]) * 0.5
    h36m_output[:,:,9,:]  = coco_input[:,:,0,:]
    h36m_output[:,:,10,:] = (coco_input[:,:,1,:] + coco_input[:,:,2,:]) * 0.5
    h36m_output[:,:,11,:] = coco_input[:,:,5,:]
    h36m_output[:,:,12,:] = coco_input[:,:,7,:]
    h36m_output[:,:,13,:] = coco_input[:,:,9,:]
    h36m_output[:,:,14,:] = coco_input[:,:,6,:]
    h36m_output[:,:,15,:] = coco_input[:,:,8,:]
    h36m_output[:,:,16,:] = coco_input[:,:,10,:]

    if is_image:
        h36m_output = h36m_output.reshape(N,J,C)
    if is_torch:
        h36m_output = torch.from_numpy(h36m_output)

    return h36m_output

if __name__ == "__main__":
    pass