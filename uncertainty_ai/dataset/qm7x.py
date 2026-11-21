import os.path as osp
import numpy as np
from typing import Callable, List, Optional
import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import QM9
from tqdm.auto import tqdm

class QM7X(InMemoryDataset):
    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 minima: str = "global",
                 target: str = 'y',
                 #full_dataset: bool = False
                 **kwargs
                 ):

        self.minima = minima
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.target = target

    @property
    def raw_file_names(self) -> List[str]:
        # return ['1000.hdf5']
    
        return ['1000.hdf5', '2000.hdf5', '3000.hdf5', '4000.hdf5', '5000.hdf5', '6000.hdf5', '7000.hdf5', '8000.hdf5']

    @property
    def processed_file_names(self) -> List[str]:
        if self.minima == "all":
            return ['full_data.pt']
        elif self.minima == "local":
            return ['local_minima_data.pt']
        elif self.minima == "global":
            return ['global_minima_data.pt']
        elif self.minima == "non-equilibrium":
            return ['non_equilibrium_data.pt']
        else:
            raise ValueError("minima must be one of 'all', 'local', or 'global'.")
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
            for molid in tqdm(mol_ids):
                ## get IDs of individual configurations/conformations of molecule
                conf_ids = list(fMOL[molid].keys())
                conf2energy = {}
                min_energy = 1e10
                # shuffle the conf_ids
                np.random.shuffle(conf_ids)
                for i, conf_id in enumerate(conf_ids):
                    if self.minima == "all":
                        ## get atomic positions and numbers and add to molecules buffer
                        pos = np.array(fMOL[molid][conf_id]['atXYZ'])
                        z = np.array(fMOL[molid][conf_id]['atNUM'])

                        ## get quantum mechanical properties and add them to properties buffer
                        energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                        force = np.array(fMOL[molid][conf_id]['totFOR'])
                        dp = np.array(fMOL[molid][conf_id]['vDIP'])
                        tq = np.array(fMOL[molid][conf_id]['vTQ'])
                        iq = np.array(fMOL[molid][conf_id]['vIQ'])
                        eq = np.array(fMOL[molid][conf_id]['vEQ'])
                        z = torch.from_numpy(z).to(torch.long)
                        # Center the positions to the center of mass
                        pos = pos - np.mean(pos, axis=0, keepdims=True)
                        pos = torch.from_numpy(pos).to(torch.float)
                        energy = torch.from_numpy(energy).to(torch.float)
                        force = torch.from_numpy(force).to(torch.float)
                        dp = torch.from_numpy(dp).to(torch.float).view(1, 3)
                        tq = torch.from_numpy(tq).to(torch.float).view(1, 3)
                        iq = torch.from_numpy(iq).to(torch.float).view(1, 3)
                        eq = torch.from_numpy(eq).to(torch.float).view(1, 3)

                        data = Data(z=z, pos=pos, energy=energy, force=force, dp = dp, atomic_numbers = z, natoms = pos.shape[0], 
                                    tq = tq, iq = iq, eq = eq,
                                    y = energy, tags = torch.ones(z.size(0), dtype=torch.long))
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        data_list.append(data)
                    elif self.minima == "non-equilibrium":
                        if conf_id[-3:] == 'opt':
                            continue
                        pos = np.array(fMOL[molid][conf_id]['atXYZ'])
                        z = np.array(fMOL[molid][conf_id]['atNUM'])

                        ## get quantum mechanical properties and add them to properties buffer
                        energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                        force = np.array(fMOL[molid][conf_id]['totFOR'])
                        dp = np.array(fMOL[molid][conf_id]['vDIP'])
                        tq = np.array(fMOL[molid][conf_id]['vTQ'])
                        iq = np.array(fMOL[molid][conf_id]['vIQ'])
                        eq = np.array(fMOL[molid][conf_id]['vEQ'])
                        dp = torch.from_numpy(dp).to(torch.float).view(1, 3)
                        tq = torch.from_numpy(tq).to(torch.float).view(1, 3)
                        iq = torch.from_numpy(iq).to(torch.float).view(1, 3)
                        eq = torch.from_numpy(eq).to(torch.float).view(1, 3)

                        if np.abs(force).min() == 0.0:
                            continue
                        z = torch.from_numpy(z).to(torch.long)
                        # Center the positions to the center of mass
                        pos = pos - np.mean(pos, axis=0, keepdims=True)
                        pos = torch.from_numpy(pos).to(torch.float)
                        energy = torch.from_numpy(energy).to(torch.float)
                        force = torch.from_numpy(force).to(torch.float)

                        data = Data(z=z, pos=pos, energy=energy, force=force, dp = dp, 
                                    tq = tq, iq = iq, eq = eq,
                                    atomic_numbers = z, natoms = pos.shape[0], y = energy, tags = torch.ones(z.size(0), dtype=torch.long))
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        data_list.append(data)
                        # only select 1 non-equilibrium conformation
                        break
                    elif self.minima == "local":
                        if conf_id[-3:] != 'opt':
                            continue
                        ## get atomic positions and numbers and add to molecules buffer
                        pos = np.array(fMOL[molid][conf_id]['atXYZ'])
                        z = np.array(fMOL[molid][conf_id]['atNUM'])

                        ## get quantum mechanical properties and add them to properties buffer
                        energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                        force = np.array(fMOL[molid][conf_id]['totFOR'])
                        dp = np.array(fMOL[molid][conf_id]['vDIP'])
                        tq = np.array(fMOL[molid][conf_id]['vTQ'])
                        iq = np.array(fMOL[molid][conf_id]['vIQ'])
                        eq = np.array(fMOL[molid][conf_id]['vEQ'])
                        dp = torch.from_numpy(dp).to(torch.float).view(1, 3)
                        tq = torch.from_numpy(tq).to(torch.float).view(1, 3)
                        iq = torch.from_numpy(iq).to(torch.float).view(1, 3)
                        eq = torch.from_numpy(eq).to(torch.float).view(1, 3)

                        if np.abs(force).min() == 0.0:
                            continue
                        z = torch.from_numpy(z).to(torch.long)
                        # Center the positions to the center of mass
                        pos = pos - np.mean(pos, axis=0, keepdims=True)
                        pos = torch.from_numpy(pos).to(torch.float)
                        energy = torch.from_numpy(energy).to(torch.float)
                        force = torch.from_numpy(force).to(torch.float)

                        data = Data(z=z, pos=pos, energy=energy, force=force, dp = dp, 
                                    tq = tq, iq = iq, eq = eq,
                                    atomic_numbers = z, natoms = pos.shape[0], y = energy, tags = torch.ones(z.size(0), dtype=torch.long))
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        data_list.append(data)
                    elif self.minima == "global":
                        if conf_id[-3:] != 'opt':
                            continue
                        energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                        if energy > min_energy:
                            continue
                        else:
                            ## get atomic positions and numbers and add to molecules buffer
                            pos = np.array(fMOL[molid][conf_id]['atXYZ'])
                            z = np.array(fMOL[molid][conf_id]['atNUM'])

                            ## get quantum mechanical properties and add them to properties buffer
                            energy = np.array(fMOL[molid][conf_id]['eMBD']) - np.array(fMOL[molid][conf_id]['eAT'])
                            force = np.array(fMOL[molid][conf_id]['totFOR'])
                            dp = np.array(fMOL[molid][conf_id]['vDIP'])
                            tq = np.array(fMOL[molid][conf_id]['vTQ'])
                            iq = np.array(fMOL[molid][conf_id]['vIQ'])
                            eq = np.array(fMOL[molid][conf_id]['vEQ'])
                            z = torch.from_numpy(z).to(torch.long)
                            # Center the positions to the center of mass
                            pos = pos - np.mean(pos, axis=0, keepdims=True)
                            pos = torch.from_numpy(pos).to(torch.float)
                            energy = torch.from_numpy(energy).to(torch.float)
                            force = torch.from_numpy(force).to(torch.float)
                            dp = torch.from_numpy(dp).to(torch.float).view(1, 3)
                            tq = torch.from_numpy(tq).to(torch.float).view(1, 3)
                            iq = torch.from_numpy(iq).to(torch.float).view(1, 3)
                            eq = torch.from_numpy(eq).to(torch.float).view(1, 3)

                            data = Data(z=z, pos=pos, energy=energy, force=force, dp = dp, 
                                        tq=tq, iq = iq, eq = eq, 
                                        atomic_numbers = z, natoms = pos.shape[0], y = energy, tags = torch.ones(z.size(0), dtype=torch.long))
                
                if self.minima == "global":
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)


        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({len(self)}, name='{self.name}')"


class QM7xDPAligned(QM7X):
    """QM7X dataset with dipoles deterministically aligned to a target plane.

    This class reuses the processed `.pt` files from `QM7X` and writes a second
    processed cache with rigidly rotated samples. The dipole is projected into
    the requested plane and a small fixed out-of-plane component is added so it
    is near-zero but not exactly zero.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        minima: str = "global",
        target: str = "dp",
        plane=(1.0, 1.0, 0.0),
        noise_std: float = 0.2,
        **kwargs,
    ):
        self.plane = torch.as_tensor(plane, dtype=torch.float32)
        self.noise_std = noise_std
        self.target = target
        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            minima=minima,
            target=target,
            **kwargs,
        )

    @property
    def processed_file_names(self) -> List[str]:
        plane_tag = "".join(str(int(v)) for v in self.plane.tolist())
        noise_str = str(self.noise_std).replace(".", "_")
        if self.minima == "all":
            return [f"full_dp_aligned_{plane_tag}_{noise_str}.pt"]
        elif self.minima == "local":
            return [f"local_minima_dp_aligned_{plane_tag}_{noise_str}.pt"]
        elif self.minima == "global":
            return [f"global_minima_dp_aligned_{plane_tag}_{noise_str}.pt"]
        elif self.minima == "non-equilibrium":
            return [f"non_equilibrium_dp_aligned_{plane_tag}_{noise_str}.pt"]
        else:
            raise ValueError("minima must be one of 'all', 'local', 'global', or 'non-equilibrium'.")

    def _base_processed_path(self) -> str:
        if self.minima == "all":
            filename = "full_data.pt"
        elif self.minima == "local":
            filename = "local_minima_data.pt"
        elif self.minima == "global":
            filename = "global_minima_data.pt"
        elif self.minima == "non-equilibrium":
            filename = "non_equilibrium_data.pt"
        else:
            raise ValueError("minima must be one of 'all', 'local', 'global', or 'non-equilibrium'.")
        return osp.join(self.processed_dir, filename)

    @staticmethod
    def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return v / torch.clamp(torch.norm(v), min=eps)

    @staticmethod
    def _rotation_from_a_to_b(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = QM7xDPAligned._safe_normalize(a)
        b = QM7xDPAligned._safe_normalize(b)

        v = torch.cross(a, b, dim=0)
        c = torch.dot(a, b)
        s = torch.norm(v)

        I = torch.eye(3, dtype=a.dtype, device=a.device)

        if s < 1e-8 and c > 0:
            return I

        if s < 1e-8 and c < 0:
            aux = torch.tensor([1.0, 0.0, 0.0], dtype=a.dtype, device=a.device)
            if torch.abs(torch.dot(a, aux)) > 0.9:
                aux = torch.tensor([0.0, 1.0, 0.0], dtype=a.dtype, device=a.device)
            axis = aux - torch.dot(aux, a) * a
            axis = QM7xDPAligned._safe_normalize(axis)
            K = torch.tensor([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0]
            ], dtype=a.dtype, device=a.device)
            return I + 2.0 * (K @ K)

        K = torch.tensor([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0]
        ], dtype=a.dtype, device=a.device)
        return I + K + K @ K * ((1.0 - c) / (s ** 2))

    def _rotation_from_dp(self, dp: torch.Tensor) -> torch.Tensor:
        d = dp.squeeze()
        d_norm = torch.norm(d)
        if d_norm < 1e-3:
            return torch.eye(3, dtype=dp.dtype, device=dp.device)

        d_unit = d / d_norm
        plane = self.plane.to(device=dp.device, dtype=dp.dtype)
        in_plane = (plane.abs() > 1e-3).to(dtype=dp.dtype)

        target = d_unit * in_plane
        target_norm = torch.norm(target)

        if target_norm < 1e-3:
            fallback = plane.clone()
            if torch.norm(fallback) < 1e-3:
                raise ValueError("plane must contain at least one non-zero entry")
            target = fallback / torch.norm(fallback)
        else:
            target = target / target_norm

        out_of_plane = 1.0 - in_plane
        target = target + out_of_plane * self.noise_std
        target = self._safe_normalize(target)
        return self._rotation_from_a_to_b(d_unit, target)

    @staticmethod
    def _rotate_sample(data: Data, R: torch.Tensor) -> Data:
        data = data.clone()
        data.pos = data.pos @ R.t()
        data.force = data.force @ R.t()
        data.dp = (data.dp.squeeze() @ R.t()).unsqueeze(0)
        data.y = data.dp
        if hasattr(data, 'tq'):
            data.tq = (data.tq.squeeze() @ R.t()).unsqueeze(0)
        if hasattr(data, 'iq'):
            data.iq = (data.iq.squeeze() @ R.t()).unsqueeze(0)
        if hasattr(data, 'eq'):
            data.eq = (data.eq.squeeze() @ R.t()).unsqueeze(0)
        return data

    def process(self):
        base_path = self._base_processed_path()
        if not osp.exists(base_path):
            QM7X(root=self.root, minima=self.minima)

        base_dataset = QM7X(root=self.root, minima=self.minima, target=self.target)
        data_list = []
        for idx in tqdm(range(len(base_dataset)), desc=f"Aligning QM7X {self.minima}"):
            data = base_dataset[idx]
            R = self._rotation_from_dp(data.dp)
            data = self._rotate_sample(data, R)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class DPRotatedDataset(Dataset):
    """Apply a fixed rigid rotation to a dataset.

    This is intended for deterministic test-time transforms after the aligned
    dataset has already been cached to disk.
    """

    def __init__(self, dataset, rotation_matrix: Optional[torch.Tensor] = None):
        self.dataset = dataset
        if rotation_matrix is None:
            # Rotate +90 degrees around y: xy plane -> yz plane.
            rotation_matrix = torch.tensor([
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=torch.float32)
        self.R = rotation_matrix.float()

    def __getitem__(self, index):
        data = self.dataset[index].clone()
        data.pos = data.pos @ self.R.t()
        data.force = data.force @ self.R.t()
        data.dp = (data.dp.squeeze() @ self.R.t()).unsqueeze(0)
        if hasattr(data, 'tq'):
            data.tq = (data.tq.squeeze() @ self.R.t()).unsqueeze(0)
        if hasattr(data, 'iq'):
            data.iq = (data.iq.squeeze() @ self.R.t()).unsqueeze(0)
        if hasattr(data, 'eq'):
            data.eq = (data.eq.squeeze() @ self.R.t()).unsqueeze(0)
        # Sync data.y with the rotated target (QM7X.get sets data.y = data.dp)
        if hasattr(data, 'y') and data.y is not None and data.y.shape == data.dp.shape:
            data.y = data.dp
        return data

    def __len__(self):
        return len(self.dataset)


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
