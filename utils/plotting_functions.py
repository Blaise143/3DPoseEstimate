import torch
import matplotlib.pyplot as plt
from typing import List

kp_array1 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_toe', 'right_toe',
                 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel']
print(len(kp_array1))
print(26*3)
exit()

def plot_keypoints(data: torch.tensor, kind: str = 'J2', title: str = "2D plot") -> None:
    """
    plots 2d keypoints. J1 represents Abi and Frank frames. J2 represents the testset.
    data is expected to be of shape (kp, 2), where kp are the joints and 2 is the x and y coordinate at the joint

    Parameters
    ----------
    data : torch.tensor
        A list of tuples corresponding to x, y coordinates
    kind:
        One of J1 or J2. J1 corresponds to Frank or Abi frames, J2 corresponds to the test set

    Args:
        title: The title of the plot
    """
    assert kind in ["J1", "J2", "mocap"], "kind should be valid"

    kp_array1 = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_toe', 'right_toe',
                 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel']

    kp_array2 = ["nose", "neck", "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
                 "right_elbow", "right_wrist", "hip", "left_hip", "left_knee", "left_ankle", "right_hip",
                 "right_knee", "right_ankle", "left_eye", "right_eye", "left_ear", "right_ear"]
    connec1 = [('neck', 'hip'), ('neck', 'head'), ('head', 'left_eye'), ('head', 'right_eye'),
               ('left_eye', 'left_ear'), ('right_eye',
                                          'right_ear'), ('left_shoulder', 'neck'),
               ('right_shoulder', 'neck'), ('left_shoulder', 'left_elbow'),
               ('right_shoulder', 'right_elbow'), ('left_elbow', 'left_wrist'),
               ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
               ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
               ('right_hip', 'right_knee'), ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
               ('right_ankle', 'right_heel'), ("left_ankle", "left_heel"), ("right_heel", "right_toe"),
               ("left_heel", "left_toe"), ("left_toe", "left_small_toe"), ("right_toe", "right_small_toe")]
    connec1_left = [
        ('head', 'left_eye'), ('left_eye', 'left_ear'), ('left_shoulder', 'neck'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),('left_shoulder', 'left_hip'),
        ('left_hip', 'hip'), ('left_hip', 'left_knee'),('left_knee', 'left_ankle'),
        ("left_ankle", "left_heel"),("left_heel", "left_toe"), ("left_toe", "left_small_toe")
    ]
    connec1_neutral = [
        ('neck', 'hip'), ('nose', 'neck')
    ]
    connec1_right = [
       ('head', 'right_eye'), ('right_eye','right_ear'), ('right_shoulder', 'neck'),
       ('right_shoulder', 'right_elbow'),('right_elbow', 'right_wrist'), ('right_shoulder', 'right_hip'),
       ('right_hip', 'hip'),('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
       ('right_ankle', 'right_heel'),("right_heel", "right_toe"),("right_toe", "right_small_toe")
    ]

    connec2 = [('neck', 'hip'), ("nose", "neck"), ("left_eye", "nose"), ("right_eye", "nose"),
               ("left_ear", "left_eye"), ("right_ear",
                                          "right_eye"), ("left_ear", "nose"),
               ("right_ear", "nose"), ('left_shoulder',
                                       'right_shoulder'), ('left_shoulder', 'left_elbow'),
               ('right_shoulder', 'right_elbow'), ('left_elbow',
                                                   'left_wrist'), ('right_elbow', 'right_wrist'),
               ('left_shoulder', 'left_hip'), ('right_shoulder',
                                               'right_hip'), ('left_hip', 'right_hip'),
               ('left_hip', 'left_knee'), ('right_hip', 'right_knee'), ('left_knee', 'left_ankle'),
               ('right_knee', 'right_ankle')]
    connec2_neutral = [
        ("neck", "hip"), ('nose', 'neck')
    ]
    connec2_left = [
        ('left_eye', 'nose'), ('left_ear', 'left_eye'), ('left_ear',
                                                         'nose'), ('left_shoulder', 'neck'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'), ('left_shoulder',
                                       'left_hip'), ('left_hip', 'hip'), ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle')
    ]
    connec2_right = [
        ('neck', 'right_shoulder'), ('right_eye', 'nose'), ('right_ear',
                                                            'right_eye'), ("right_ear", "nose"),
        ("right_shoulder", "right_elbow"),
        ('hip', 'right_hip'), ("right_elbow", "right_wrist"), ("right_shoulder",
                                                               "right_hip"), ("right_hip", "right_knee"),
        ("right_knee", "right_ankle")
    ]

    keypoint_coordinates = [(float(x[0]), float(x[1])) for x in data]
    print(f"kp coordinates: {len(keypoint_coordinates)}")

    if kind == "J1":
        keypoint_array = kp_array1
        connections = connec1

    else:
        keypoint_array = kp_array2
        connections = connec2

    keypoint_dict = dict(zip(keypoint_array, keypoint_coordinates))

    fig, ax = plt.subplots()
    print(f"dict: {keypoint_dict}")
    for keypoint in keypoint_array:
        x, y = keypoint_dict[keypoint]
        ax.scatter(x, y, label=keypoint, marker=".")

    # Add labels for each keypoint
    for keypoint in keypoint_array:
        x, y = keypoint_dict[keypoint]
        ax.scatter(x, y, label=keypoint, marker=".")  # TODO: CHECK THIS
        ax.text(x + 0.02, y + 0.02, keypoint,
                fontsize=9, color='black')  # AND THIS

    def connect(connections, color, ax=ax):
        for start, end in connections:
            x1, y1 = keypoint_dict[start]
            x2, y2 = keypoint_dict[end]
            ax.plot([x1, x2], [y1, y2], color)

    if kind != "J1":
        connect(connec2_neutral, color='black')
        connect(connec2_left, color="blue")
        connect(connec2_right, color="red")
    else:
        connect(connec1_neutral, color='black')
        connect(connec1_left, color="blue")
        connect(connec1_right, color="red")

    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert the y-axis to match typical image coordinates
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid()
    # plt.gca().set_aspect("equal")
    # setting figsize
    # plt.rcParams["figure.figsize"] = (18, 10)
    return fig

def plot_J1_overlay(
        data: torch.tensor,
        mirror_data: torch.Tensor,
        image_path: str,
        title: str = "2D plot",
):
    """
    Plots the 2D keypoints on top of the frames
    Args:
        data: tensor of shape (N, 2) where N is the number of keypoints
        mirror_data: the same but fot the mirror person
        image_path: The path to the corresponding image
        title: the title o

    Returns:
    The fig object.. displays when we call plt.show()
    """
    keypoint_array = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
                 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
                 'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck', 'hip', 'left_toe', 'right_toe',
                 'left_small_toe', 'right_small_toe', 'left_heel', 'right_heel']


    left_connections = [
        ('head', 'left_eye'), ('left_eye', 'left_ear'), ('left_shoulder', 'neck'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'), ('left_shoulder', 'left_hip'),
        ('left_hip', 'hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ("left_ankle", "left_heel"), ("left_heel", "left_toe"), ("left_toe", "left_small_toe")
    ]
    neutral_connections = [
        ('neck', 'hip'), ('nose', 'neck')
    ]
    right_connections = [
        ('head', 'right_eye'), ('right_eye', 'right_ear'), ('right_shoulder', 'neck'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_shoulder', 'right_hip'),
        ('right_hip', 'hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'), ("right_heel", "right_toe"), ("right_toe", "right_small_toe")
    ]

    keypoint_coordinates = [(float(x[0]), float(x[1])) for x in data]
    mirror_keypoint_coordinates = [
        (float(x[0]), float(x[1])) for x in mirror_data]

    keypoint_dict = dict(zip(keypoint_array, keypoint_coordinates))
    print(f"dict: {keypoint_dict}")
    keypoint_mirror_dict = dict(
        zip(keypoint_array, mirror_keypoint_coordinates))


    fig, ax = plt.subplots()
    im = plt.imshow(plt.imread(image_path))
    for keypoint in keypoint_array:
        x, y = keypoint_dict[keypoint]
        xm, ym = keypoint_mirror_dict[keypoint]  # TODO: Ensure this is fine
        ax.scatter(x, y, label=keypoint, marker=".")
        ax.scatter(xm, ym, label=keypoint, marker=".")

    def connect_points(connections: List[tuple], color: str):
        for start, end in connections:
            x1, y1 = keypoint_dict[start]
            x2, y2 = keypoint_dict[end]

            x1_m, y1_m = keypoint_mirror_dict[start]
            x2_m, y2_m = keypoint_mirror_dict[end]
            ax.plot([x1,x2], [y1, y2], color)
            ax.plot([x1_m, x2_m], [y1_m, y2_m], color)

    connect_points(left_connections, "blue")
    connect_points(right_connections, "red")
    connect_points(neutral_connections, "black")
    # ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    plt.gca().set_aspect("equal")
    return fig



def obtain_data(path: str = "../zju-m-seq1/annots/3") -> tuple:
    """
    Obtain data from the specified path. The default path is the testset. 

    Parameters
    ----------
    path : str
        path to the data

    Returns
    -------
    tuple
        the tuple of the tensors, h for humans, r for their reflections
    """

    h, r = [], []
    for i in sorted(os.listdir(path)):
        f = open(path + "/" + i)
        item = json.load(f)
        h.append(item["annots"][0]["keypoints"])
        r.append(item["annots"][1]["keypoints"])
    h, r = torch.tensor(h)[:, :, :2], torch.tensor(r)[:, :, :2]

    return h, r
