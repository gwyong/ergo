from abc import ABC, abstractmethod
import numpy as np


class ErgoAngleConverter(ABC):
    """
    Convert different human pose format to key ergo angles
    Accepted formats: H36M-17, SMPL-24, VEHS-ergo-66
    Output formats: REBA, RULA, (OWAS)
    """
    def __init__(self):
        self.pose = None
        pass

    @abstractmethod
    def load_pose(self, pose):
        pass

    def output_reba(self):
        '''implement here'''
        pass

    def output_rula(self):
        pass

    def output_owas(self):
        pass


class H36MConverter(ErgoAngleConverter):
    def __init__(self):
        super().__init__()
        pass

    def load_pose(self, pose):
        '''implement here'''
        pass

class SMPLConverter(ErgoAngleConverter):
    def __init__(self):
        super().__init__()
        pass

    def load_pose(self, pose):
        '''implement here'''
        pass



if __name__ == "__main__":
    ''' Test & example here'''
    sample_pose = np.array([[0.08533354, 1.03611605, 0.09013124],
                            [0.15391247, 0.91162637, -0.00353906],
                            [0.22379057, 0.87361878, 0.11541229],
                            [0.4084777, 0.69462843, 0.1775224],
                            [0.31665226, 0.46389668, 0.16556387],
                            [0.1239769, 0.82994377, -0.11715403],
                            [0.08302169, 0.58146328, -0.19830338],
                            [-0.06767788, 0.53928527, -0.00511249],
                            [0.11368726, 0.49372503, 0.21275574],
                            [0.069179, 0.07140968, 0.26841402],
                            [0.10831762, -0.36339359, 0.34032449],
                            [0.11368726, 0.41275504, -0.01171348],
                            [0., 0., 0.],
                            [0.02535541, -0.43954643, 0.04373671],
                            [0.26709431, 0.33643749, 0.17985192],
                            [-0.15117603, 0.49462711, 0.02703403],
                            [-0.15117603, 0.49462711, 0.02703403]
                            ])

    # h36mConverter = H36MConverter()
    # h36mConverter.load_pose(sample_pose)
    # h36mConverter.output_reba()