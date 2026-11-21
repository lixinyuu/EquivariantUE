import numpy as np
from typing import Callable, List, Optional
import h5py
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import QM9


class QM7X(InMemoryDataset):
    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 is_equilibrium: bool = True,
                 target: str = 'y',
                 #full_dataset: bool = False
                 **kwargs
                 ):
        self.is_equilibrium = is_equilibrium
        if self.is_equilibrium == 'both':
            self.full_dataset = True
        else:
            self.full_dataset = False
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.target = target

    @property
    def raw_file_names(self) -> List[str]:
        return ['1000.hdf5', '2000.hdf5', '3000.hdf5', '4000.hdf5', '5000.hdf5', '6000.hdf5', '7000.hdf5', '8000.hdf5']

    @property
    def processed_file_names(self) -> List[str]:
        if self.full_dataset:
            return ['full_data.pt']
        else:
            if self.is_equilibrium:
                return ['data.pt']
            else:
                return ['non_equilibrium_data.pt']
    
    def get(self, idx: int) -> Data:
        data = super().get(idx)
        # get the self.target attribute of data
        if hasattr(data, self.target):
            target = getattr(data, self.target)
        else:
            raise ValueError(f"Target {self.target} not found in data.")
        data.y = target
        return data
    
    def process(self):
        data_list = []
        for raw_path in self.raw_paths:
            fMOL = h5py.File(raw_path, 'r')
            mol_ids = list(fMOL.keys())
            for molid in mol_ids:
                ## get IDs of individual configurations/conformations of molecule
                conf_ids = list(fMOL[molid].keys())
                for i, conf_id in enumerate(conf_ids):
                    if self.full_dataset:
                        ## get atomic positions and numbers and add to molecules buffer
                        pos = np.array(fMOL[molid][conf_id]['atXYZ'])
                        z = np.array(fMOL[molid][conf_id]['atNUM'])

                        ## get quantum mechanical properties and add them to properties buffer
                        energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                        force = np.array(fMOL[molid][conf_id]['totFOR'])
                        dp = np.array(fMOL[molid][conf_id]['vDIP'])
                    else:
                        if conf_id[-3:] != 'opt':
                            continue
                        if self.is_equilibrium:
                            ## get atomic positions and numbers and add to molecules buffer
                            pos = np.array(fMOL[molid][conf_id]['atXYZ'])
                            z = np.array(fMOL[molid][conf_id]['atNUM'])

                            ## get quantum mechanical properties and add them to properties buffer
                            energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                            force = np.array(fMOL[molid][conf_id]['totFOR'])
                            dp = np.array(fMOL[molid][conf_id]['vDIP'])
                            if np.abs(force).min() == 0.0:
                                continue
                        else:
                            assert conf_ids[i-1][-3:] != 'opt'
                            ## get atomic positions and numbers and add to molecules buffer
                            pos = np.array(fMOL[molid][conf_ids[i-1]]['atXYZ'])
                            z = np.array(fMOL[molid][conf_ids[i-1]]['atNUM'])

                            ## get quantum mechanical properties and add them to properties buffer
                            energy = np.array(fMOL[molid][conf_ids[i-1]]['eMBD']) - np.array(fMOL[molid][conf_ids[i-1]]['eAT'])
                            force = np.array(fMOL[molid][conf_ids[i-1]]['totFOR'])
                            dp = np.array(fMOL[molid][conf_id]['vDIP'])

                        

                    z = torch.from_numpy(z).to(torch.long)
                    # Center the positions to the center of mass
                    pos = pos - np.mean(pos, axis=0, keepdims=True)
                    pos = torch.from_numpy(pos).to(torch.float)
                    energy = torch.from_numpy(energy).to(torch.float)
                    force = torch.from_numpy(force).to(torch.float)
                    dp = torch.from_numpy(dp).to(torch.float)

                    data = Data(z=z, pos=pos, energy=energy, force=force, dp = dp, atomic_numbers = z, natoms = pos.shape[0], y = energy, tags = torch.ones(z.size(0), dtype=torch.long))
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"

atomref = torch.zeros(100)
atomref[torch.tensor([1, 6, 7, 8, 9])] = torch.tensor([
        -13.61312172, -1029.86312267, -1485.30251237, -2042.61123593,
        -2713.48485589
    ])
class QM9HelpTransform:
    def __call__(self, data):
        data.force = torch.zeros_like(data.pos)
        data.atomic_numbers = data.x[:, 5]
        data.natoms = data.pos.size(0)
        data.y = data.y[:, [7]] - (atomref[data.atomic_numbers.long()]).sum()
        return data


class QM9P(QM9):
    def __init__(self, root, transform=None, pre_transform=None):
        if transform is None:
            transform = QM9HelpTransform()
        super().__init__(root, transform=transform, pre_transform=pre_transform)


