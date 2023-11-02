import torch
import numpy as np
import cv2

def torch_image_to_numpy_frames(x):
    is_torch, is_image = False, False
    if isinstance(x, torch.Tensor):
        x = x.numpy()
        is_torch = True
    
    if x.ndim == 3:
        N, J, C = x.shape
        x = x.reshape(N, 1, J, C) # Assume the image as a single frame
        is_image = True
    return is_torch, is_image, x

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

    is_torch, is_image, coco_input = torch_image_to_numpy_frames(coco_input)

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

def draw_2d_pose_from_image(image, keypoints, filename):
    output = cv2.imread(image)

    connections = [(1, 2), (2, 3), (4, 5), (5, 6), (0, 1), (0, 4),
                   (0, 7), (7, 8), (8, 9), (9, 10),                  
                   (8, 11), (11, 12), (12, 13),                      
                   (8, 14), (14, 15), (15, 16)]

    # Draw keypoints on the image
    for person_keypoints in keypoints:
        for keypoint in person_keypoints:
            x, y, visible = keypoint
            if visible:  # Only draw visible keypoints
                cv2.circle(output, (int(x), int(y)), 5, (255, 0, 0), -1)  # Use colors for keypoints

    # Draw lines connecting keypoints
    for person_keypoints in keypoints:
        for connection in connections:
            start_point = person_keypoints[connection[0]][:2]
            end_point = person_keypoints[connection[1]][:2]
            cv2.line(output, (int(start_point[0]), int(start_point[1])),
                    (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)  # Green lines for connections (B, G, R)

    cv2.imwrite(filename, output)

def h36m2rcoco(h36m_input):
    """
    H36M to relational COCO format for ergonomic risk assessment
    Output:
        rcoco_input index
        [0] = Head
        [1] = Nose
        [2, 3, 4, 14]: Left Shoulder, Elbow, Wrist + Hand (optional)
        [5, 6, 7, 15]: Right Shoulder, Elbow, Wrist + Hand (optional)
        [8, 9, 10]: Left Hip, Knee, Ankle
        [11, 12, 13]: Right Hip, Knee, Ankle
    """
    is_torch, is_image, h36m_input = torch_image_to_numpy_frames(h36m_input)
    N, T, _, C = h36m_input.shape
    J = 14 # when using hands, please change 14 to 16.
    coco_output = np.zeros((N, T, J, C)) 
    rcoco_output = np.zeros((N, T, J, C))
    
    # Mapping from H36M to COCO
    coco_output[:,:,0,:] = h36m_input[:,:,10,:]
    coco_output[:,:,1,:] = h36m_input[:,:,8,:]
    coco_output[:,:,2,:] = h36m_input[:,:,11,:]
    coco_output[:,:,3,:] = h36m_input[:,:,12,:]
    coco_output[:,:,4,:] = h36m_input[:,:,13,:]
    coco_output[:,:,5,:] = h36m_input[:,:,14,:]
    coco_output[:,:,6,:] = h36m_input[:,:,15,:]
    coco_output[:,:,7,:] = h36m_input[:,:,16,:]
    coco_output[:,:,8,:] = h36m_input[:,:,4,:]
    coco_output[:,:,9,:] = h36m_input[:,:,5,:]
    coco_output[:,:,10,:] = h36m_input[:,:,6,:]
    coco_output[:,:,11,:] = h36m_input[:,:,1,:]
    coco_output[:,:,12,:] = h36m_input[:,:,2,:]
    coco_output[:,:,13,:] = h36m_input[:,:,3,:]
    # coco_output[:,:,14,:] left hand (optional)
    # coco_output[:,:,15,:] right hand (optional)
    
    root_points = (h36m_input[:,:,1,:]+h36m_input[:,:,4,:])*0.5
    root_points = np.expand_dims(root_points, axis=2)
    root_points = np.concatenate([root_points] * 14, axis=2)
    rcoco_output = rcoco_output-root_points

    if is_image:
        rcoco_output = rcoco_output.reshape(N,J,C)
    if is_torch:
        rcoco_output = torch.from_numpy(rcoco_output)
    return rcoco_output

if __name__ == "__main__":
    pass