import torch
import matplotlib.pyplot as plt


def plot_keypoints(data: torch.tensor, kind: str = 'J2', title: str = "2D plot") -> None:
    """
    A function that plots 2d keypoints. J1 represents Abi and Frank frames. J2 represents the testset.
    data is expected to be of shape (kp, 2), where kp are the joints and 2 is the x and y coordinate at the joint

    Parameters
    ----------
    data : torch.tensor
        A list of tuples corresponding to x, y coordinates
    kind:
        One of J1 or J2. J1 corresponds to Frank or Abi frames, J2 corresponds to the test set
    """
    assert kind in ["J1", "J2"], "kind should be valid"

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
               ('right_shoulder', 'neck'), ('left_shoulder',
                                            'left_elbow'), ('right_shoulder', 'right_elbow'),
               ('left_elbow', 'left_wrist'), ('right_elbow',
                                              'right_wrist'), ('left_shoulder', 'left_hip'),
               ('right_shoulder', 'right_hip'), ('left_hip',
                                                 'right_hip'), ('left_hip', 'left_knee'),
               ('right_hip', 'right_knee'), ('left_knee',
                                             'left_ankle'), ('right_knee', 'right_ankle'),
               ('right_ankle', 'right_heel')]
    connec2 = [('neck', 'hip'), ("nose", "neck"), ("left_eye", "nose"), ("right_eye", "nose"),
               ("left_ear", "left_eye"), ("right_ear",
                                          "right_eye"), ("left_ear", "nose"),
               ("right_ear", "nose"), ('left_shoulder',
                                       'right_shoulder'), ('left_shoulder', 'left_elbow'),
               ('right_shoulder', 'right_elbow'), ('left_elbow',
                                                   'left_wrist'), ('right_elbow', 'right_wrist'),
               ('left_shoulder', 'left_hip'), ('right_shoulder',
                                               'right_hip'), ('left_hip', 'right_hip'),
               ('left_hip', 'left_knee'), ('right_hip', 'right_knee'), ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')]
    connec2_neutral = [
        ("neck", "hip"), ('nose', 'neck')
    ]
    connec2_left = [
        ('left_eye', 'nose'), ('left_ear', 'left_eye'), ('left_ear',
                                                         'nose'), ('left_shoulder', 'neck'), ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'), ('left_shoulder',
                                       'left_hip'), ('left_hip', 'hip'), ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle')
    ]
    connec2_right = [
        ('neck', 'right_shoulder'), ('right_eye', 'nose'), ('right_ear',
                                                            'right_eye'), ("right_ear", "nose"), ("right_shoulder", "right_elbow"),
        ('hip', 'right_hip'), ("right_elbow", "right_wrist"), ("right_shoulder",
                                                               "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")
    ]

    keypoint_coordinates = [(float(x[0]), float(x[1])) for x in data]

    if kind == "J1":
        keypoint_array = kp_array1
        connections = connec1
    else:
        keypoint_array = kp_array2
        connections = connec2
    keypoint_dict = dict(zip(keypoint_array, keypoint_coordinates))

    fig, ax = plt.subplots()
    for keypoint in keypoint_array:
        x, y = keypoint_dict[keypoint]
        ax.scatter(x, y, label=keypoint, marker=".")

    # Add labels for each keypoint
    for keypoint in keypoint_array:
        x, y = keypoint_dict[keypoint]

    for start, end in connec2_neutral:
        x1, y1 = keypoint_dict[start]
        x2, y2 = keypoint_dict[end]
        ax.plot([x1, x2], [y1, y2], 'black')

    for start, end in connec2_left:
        x1, y1 = keypoint_dict[start]
        x2, y2 = keypoint_dict[end]
        ax.plot([x1, x2], [y1, y2], 'blue')

    for start, end in connec2_right:
        x1, y1 = keypoint_dict[start]
        x2, y2 = keypoint_dict[end]
        ax.plot([x1, x2], [y1, y2], 'red')

    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert the y-axis to match typical image coordinates
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid()
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
        f = open(path+"/"+i)
        item = json.load(f)
        h.append(item["annots"][0]["keypoints"])
        r.append(item["annots"][1]["keypoints"])
    h, r = torch.tensor(h)[:, :, :2], torch.tensor(r)[:, :, :2]

    return h, r
