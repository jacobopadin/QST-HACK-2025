def extract_angles(file="angles_advection.txt"):
    cos_angles = []
    sin_angles = []

    with open(file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        seq = list(map(float, line.strip().split()))
        for i in range(0, len(seq)-1, 2):
            cos_angles.append(seq[i])
            sin_angles.append(seq[i+1])

    return cos_angles, sin_angles