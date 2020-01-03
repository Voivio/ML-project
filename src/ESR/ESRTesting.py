import util

N_aug = 20

# Load training dataset

# Create InitSet
init_set = util.CreateInitialSet(train_set, stage = 'train')

# Initialization
util.Initialization(train_set, N_aug, init_set)

#
for t in range(T):
    for i in range(N_int):

S= util.CombineMultipleResults()

return
