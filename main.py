from lraAttack import *
from linearAtt import *


####################### CONFIG ############################ 

# attack settings
SboxNum = 14
TRACES_NUMBER = 100
TRACE_LENGTH = 2000 ## number of samples
TRACE_STARTING_SAMPLE = 0
offset = 0
traceRoundNumber = 100

# visual data 


# print options


# load samle data

PATH_TRACE="traces/aes192_sb_ciph_ec40554bf67c9655d85cfd69ac04012f7ee1340ccf7b24fd.trs.npz"

sampleRange = (TRACE_STARTING_SAMPLE, TRACE_STARTING_SAMPLE + TRACE_LENGTH)
npzfile = np.load(PATH_TRACE)
traces = npzfile['traces'][offset:offset + TRACES_NUMBER,sampleRange[0]:sampleRange[1]]
data = npzfile['data'][:, SboxNum]

# attack LRA 

attackAESinRounds(data, traces, traceRoundNumber , TRACES_NUMBER, SboxNum)








# with trsfile.open('des_dec_60e6661ad826ecec.trs', 'r') as traces:

#     for header, value in traces.get_headers().items():
#         print(header, '=', value)
#     print()


#     for i, trace in enumerate(traces[0:10]):
#         for tr in trace:
#             print(tr)
#         x = np.arange(0, len(trace))
#         print(x)
#         plt.plot(x, trace)
#         print('Trace {0:d} contains {1:d} samples'.format(i, len(trace)))
#         print('  - minimum value in trace: {0:f}'.format(min(trace)))
#         print('  - maximum value in trace: {0:f}'.format(max(trace)))

#     plt.show()


