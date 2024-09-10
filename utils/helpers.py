import pickle
import h5py
import torch


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
    A = torch.eye(3) - 2 * torch.matmul(normal.view(3, 1), normal.view(1, 3))
    return A


if __name__ == "__main__":

    abi = extract_pickle(file_path="../data/iccv2023/Abi_0_1999_2000iter.pickle")
    print(abi[0].keys())
    exit()
    test_set_normal = h5py.File("../data/zju_testset_properties/v_mirror_test_h5py.h5", "r")
    print(list(test_set_normal["m_normal"])[0])
    print()

    out = extract_pickle(file_path="../data/k_matrices/ab_calibration.pickle")
    print(out)
