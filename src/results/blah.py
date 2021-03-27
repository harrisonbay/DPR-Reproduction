# generates fake results
f = open("results-test.txt", "w")
f.write("experiment version | top-k | accuracy\n")
f.write("3.0\t10\t20\n")
f.write("3.0\t100\t80\n")
f.write("3.0\t1000\t90\n")