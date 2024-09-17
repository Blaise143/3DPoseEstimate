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


def project_2d(x: torch.tensor) -> torch.tensor:
    K_matrix = torch.tensor(
        [
            [1, 0],
            [0, 1],
            [0, 0]
        ]
    )

    projection = torch.matmul(x, K_matrix)
    return projection


def obtain_K() -> torch.tensor:
    """
    outputs the orthographic projection
    """
    K = torch.tensor(
        [
            [1, 0],
            [0, 1.],
            [0, 0]
        ]
    )
    return K


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state. # Credit: I made some modifications to a some code I got from stackOverflow(dont remember the post)
    """

    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss

    def __call__(
        self, current_valid_loss,
        epoch, model, optimizer, criterion, path
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            # print(f"\nBest validation loss: {self.best_valid_loss}")
            # print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
            }, path)


if __name__ == "__main__":
    abi_pickle_path = "../data/iccv2023/Abi_0_1999_2000iter.pickle"
    frank_pickle_path = "../data/iccv2023/Frank_0_1999_2000iter.pickle"
    abi = extract_pickle(file_path=abi_pickle_path)
    frank = extract_pickle(file_path=frank_pickle_path)
    abi_normal = abi[0]['n_m'][0].detach()
    frank_normal = frank[0]['n_m'][0].detach()
    # print(abi_normal, print(frank_normal))
    A_abi = obtain_A_from_normal(abi_normal)
    A_frank = obtain_A_from_normal(frank_normal)
    print(abi[0].keys())
    # exit()
    print(abi[0]['final_A_dash'][0])
    # print(abi)
    print(frank[0]['final_A_dash'][0])
    exit()
    # print(
    #     A_abi, A_frank
    # )
    # exit()
    #***
    abi2 = abi[0]["K_optim"][0]
    frank2 = frank[0]["K_optim"][0]
    print(abi2)
    a = torch.arange(18).view(2, -1, 3)
    print(a)
    b = project_2d(a)
    print(b)
    # print(a)

    # print(A_abi)
    # print("*********************")
    # print(frank2)
    # print(A_frank)
    # print(f"A_normal: {abi_normal}\nF_normal: {frank_normal}")
    # print(abi_normal.mean(dim=0))
    # print(frank_normal.mean(dim=0))
    # print(abi_normal[0])
    # print(frank_normal[0])
    print()
    exit()
    test_set_normal = h5py.File("../data/zju_testset_properties/v_mirror_test_h5py.h5", "r")
    print(list(test_set_normal["m_normal"])[0])
    print()

    out = extract_pickle(file_path="../data/k_matrices/ab_calibration.pickle")
    print(out)
