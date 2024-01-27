import torch

Rot_3 = [4.3810293001263906e-01, -8.9827966092424538e-01,
         3.4051042335932702e-02, -5.2676777198597702e-02,
         -6.3468951146432584e-02, -9.9659261957147827e-01,
         8.9738006433077733e-01, 4.3481644749272591e-01,
         -7.5124411022619408e-02]
T_3 = [-1.3874052527583853e+00, 7.6399863909829924e-01,
       4.6984348420284645e+00]

Rot_3 = torch.tensor(Rot_3).view(-1, 3)
T_3 = torch.tensor(T_3)
