import numpy as np
import os
from Constants import ALL_GAMES


def load_file(filename):
    rows = []
    with open(filename, "r") as file:
        for line in file.readlines():
            if line.endswith("\n"):
                rows.append(line[:-1])
            else:
                if len(line) > 0:
                    rows.append(line)

    a = np.chararray((len(rows), len(rows[0])), unicode=True)
    for i, row in enumerate(rows):
        for j, c in enumerate(row):
            if j >= len(rows[0]):
                break
            a[i, j] = c
    return a


def write_file(filename, array):
    with open(filename, "w") as file:
        file.writelines(["".join(row)+"\n" for row in array[:-1, :]])
        file.write("".join(array[-1, :]))
    return


def create_additional_level_files(basefile, level, targetfolder):
    a = load_file(basefile)
    known_levels = []

    write_file(target_folder+level+"_backup.txt", a)

    rot0 = a
    rot90 = np.rot90(a)
    rot180 = np.rot90(rot90)
    rot270 = np.rot90(rot180)
    if sum([np.array_equal(level, a) for level in known_levels]) == 0:
        write_file(target_folder+level+"_01_rot0.txt", rot0)
        known_levels.append(rot0)

    if sum([np.array_equal(level, rot90) for level in known_levels]) == 0:
        write_file(target_folder+level+"_02_rot90.txt", rot90)
        known_levels.append(rot90)

    if sum([np.array_equal(level, rot180) for level in known_levels]) == 0:
        write_file(target_folder+level+"_03_rot180.txt", rot180)
        known_levels.append(rot180)

    if sum([np.array_equal(level, rot270) for level in known_levels]) == 0:
        write_file(target_folder+level+"_04_rot270.txt", rot270)
        known_levels.append(rot270)

    rot0flipx = np.flip(rot0, 0)
    rot90flipx = np.flip(rot90, 0)
    rot180flipx = np.flip(rot180, 0)
    rot270flipx = np.flip(rot270, 0)
    if sum([np.array_equal(level, rot0flipx) for level in known_levels]) == 0:
        write_file(target_folder+level+"_05_rot0flipx.txt", rot0flipx)
        known_levels.append(rot0flipx)

    if sum([np.array_equal(level, rot90flipx) for level in known_levels]) == 0:
        write_file(target_folder+level+"_06_rot90flipx.txt", rot90flipx)
        known_levels.append(rot90flipx)

    if sum([np.array_equal(level, rot180) for level in known_levels]) == 0:
        write_file(target_folder+level+"_07_rot180flipx.txt", rot180flipx)
        known_levels.append(rot180)

    if sum([np.array_equal(level, rot270flipx) for level in known_levels]) == 0:
        write_file(target_folder + level + "_08_rot270flipx.txt", rot270flipx)
        known_levels.append(rot270flipx)

    rot0flipy = np.flip(rot0, 1)
    rot90flipy = np.flip(rot90, 1)
    rot180flipy = np.flip(rot180, 1)
    rot270flipy = np.flip(rot270, 1)
    if sum([np.array_equal(level, rot0flipy) for level in known_levels]) == 0:
        write_file(target_folder+level+"_09_rot0flipy.txt", rot0flipy)
        known_levels.append(rot0flipy)
    if sum([np.array_equal(level, rot90flipy) for level in known_levels]) == 0:
        write_file(target_folder+level+"_10_rot90flipy.txt", rot90flipy)
        known_levels.append(rot90flipy)
    if sum([np.array_equal(level, rot180flipy) for level in known_levels]) == 0:
        write_file(target_folder+level+"_11_rot180flipy.txt", rot180flipy)
        known_levels.append(rot180flipy)
    if sum([np.array_equal(level, rot270flipy) for level in known_levels]) == 0:
        write_file(target_folder+level+"_12_rot270flipy.txt", rot270flipy)
        known_levels.append(rot270flipy)

    print("generated ", len(known_levels), "levels for game ", game, level)


if __name__ == "__main__":

    levels = ["lvl0", "lvl1", "lvl2", "lvl3", "lvl4"]

    for game in ALL_GAMES:
        for level in levels:
            # the path to the GVGAI framework to load the levels from
            gvgai_file = f"D:\\GVGAI_GYM\\gym_gvgai\\envs\\games\\{game}_v0\\{game}_{level}.txt"
            target_folder = f"gvgai-{game}\\"
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            create_additional_level_files(gvgai_file, level, target_folder)
