from matplotlib import pyplot as plt
import numpy as np
import trsfile 

my_trace = 0

with trsfile.open('des_dec_60e6661ad826ecec.trs', 'r') as traces:

    for header, value in traces.get_headers().items():
        print(header, '=', value)
    print()


    for i, trace in enumerate(traces[0:10]):
        for tr in trace:
            print(tr)
        x = np.arange(0, len(trace))
        print(x)
        plt.plot(x, trace)
        print('Trace {0:d} contains {1:d} samples'.format(i, len(trace)))
        print('  - minimum value in trace: {0:f}'.format(min(trace)))
        print('  - maximum value in trace: {0:f}'.format(max(trace)))
    
    plt.show()


