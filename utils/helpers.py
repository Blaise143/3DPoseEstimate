import pickle
import h5py
import torch
from sympy.physics.units.systems.si import dimex


def extract_pickle(file_path: str):
    """
    extracts from pickle file
    """
    objects = []
    with (open(file_path, "rb")) as file:
        while True:
            try:
                objects.append(pickle.load(file))
            except EOFError:
                break
    return objects


def obtain_A_from_normal(normal: torch.tensor) -> torch.tensor:
    A = torch.eye(3) - 2*torch.matmul(normal.view(3, 1), normal.view(1, 3))
    return A


if __name__ == "__main__":

    abi_pickle_path = "../data/iccv2023/Abi_0_1999_2000iter.pickle"
    frank_pickle_path = "../data/iccv2023/Frank_0_1999_2000iter.pickle"
    abi = extract_pickle(file_path=abi_pickle_path)
    frank= extract_pickle(file_path=frank_pickle_path)
    abi_normal = abi[0]['n_m'][0].detach()
    frank_normal = frank[0]['n_m'][0].detach()
    # print(abi_normal, print(frank_normal))
    A_abi = obtain_A_from_normal(abi_normal)
    A_frank = obtain_A_from_normal(frank_normal)
    print(abi[0].keys())
    # print(abi)
    print(
        A_abi, A_frank
    )
    # print(abi_normal.mean(dim=0))
    # print(frank_normal.mean(dim=0))
    # print(abi_normal[0])
    # print(frank_normal[0])
    exit()
    test_set_normal = h5py.File("../data/zju_testset_properties/v_mirror_test_h5py.h5", "r")
    print(list(test_set_normal["m_normal"])[0])
    print()

    out = extract_pickle(file_path="../data/k_matrices/ab_calibration.pickle")
    print(out)
