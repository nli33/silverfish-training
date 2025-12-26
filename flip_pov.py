# sometimes evaluation is only given from the perspective of white
# our NNUE should return evaluations from the perspective of the side to move
# example : black to move, black is winning, absolute evaluation might be -790cp but our NNUE should return +790cp

write_lines = []

with open("train.csv") as f:
    write_lines.append(f.readline()) # header
    for line in f.readlines():
        if not line.strip(): continue
        fen, cp = line.split(',')
        if "w" in fen: # WHITE
            write_lines.append(fen + "," + cp)
        else:
            write_lines.append(fen + "," + str(-int(cp.strip())) + "\n")

with open("train_flip.csv", 'w') as f:
    f.writelines(write_lines)
