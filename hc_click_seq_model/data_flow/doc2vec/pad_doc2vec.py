import sys

doc2vec_file = sys.argv[1]
lines = int(sys.argv[2])
size = int(sys.argv[3])
padding_value = float(sys.argv[4])

def doc2vec_padding(doc2vec_file, lines, size, padding_value):
  with open(doc2vec_file, "r+") as f:
    old = f.read()
    f.seek(0)
    for i in range(lines):
      for j in range(size):
        if j != (size -1):
          f.write(str(padding_value) + ' ')
	else:
	  f.write(str(padding_value))
      f.write('\n')
    f.write(old)

doc2vec_padding(doc2vec_file, lines, size, padding_value)
