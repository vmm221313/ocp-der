import lmdb
# This scripts takes one LMDB file and returns another LMDB file that is a subsplit of the original file.
# e.g. The first 100 key-value pairs in a LMDB file
# Assumes ascending order of keys in the LMDB (i.e. 0, 1, 2, 3, ... , etc.)
# Create a list of lists
# Each entry in the list is a list of key-values for each ensemble split.
def splitKeys(rangeVal, chunks):
    rangeList = range(rangeVal)
    # Avoid creating empty lists for the scenario where rangeList < chunks
    chunks = min(chunks, rangeVal)
    divisionResult, modulus = divmod(rangeVal, chunks)
    return (rangeList[ind*divisionResult + min(ind, modulus):(ind + 1)*divisionResult + min(ind + 1, modulus)] for ind in range(chunks))
# Number of subsets to split the training data into.
numSubsets = 5
# Maximum mapping size for each lmdb file that is to hold an ensemble subsplit
gigSize = 1073741824
numGigs = 100
gigPerLMDB = gigSize * numGigs
# Original LMDB to be split
origPath = "/ocean/projects/chm210001p/shared/shared_data/is2re/all/cgcnn/train/data.lmdb"
# Parent path for the LMDB splits
copyPathParent = "/ocean/projects/chm210001p/shared/shared_data/is2re/prototype/200k"
# Filename of the split, includes file extension
splitName = "data.lmdb"
env = lmdb.open(origPath, map_size = gigPerLMDB, subdir = False, readonly = True, lock = False, map_async = True, readahead = False, meminit = False)
numKeys = 200000
keyStart = 0
keyEnd = 199999
print("\n Checking to see if the starting key and ending key align with the number of keys desired: ")
if ((keyEnd - keyStart + 1) == numKeys):
    print("TRUE\n")
else:
    raise ValueError("The number of keys inclusive and between the starting key/end key do not match the number of keys desired.")
# Create a "list of lists" where each list in the list is a specification of keys for that ensemble training data split.
ensembleKeySplits = [list(range(keyStart, keyEnd + 1))]
# Count the number of keys that are supposed to be in each split.
ensembleKeySplitCounts = []
for splitInd in range(len(ensembleKeySplits)):
    # Get the amount of keys to be put in the ensemble split.
    keySplitCount = len(ensembleKeySplits[splitInd])
    ensembleKeySplitCounts.append(keySplitCount)
print("\nSplitting LMDB into a subset with the following key counts in each split:")
print(ensembleKeySplitCounts)
print("\nLMDB save directory:")
print(copyPathParent)
with env.begin() as txn_old:
        for splitInd in range(len(ensembleKeySplits)):
            # Set up the path/filename of new LMDB
            copyPath = copyPathParent + "/" + splitName
            cop = lmdb.open(copyPath, map_size = gigPerLMDB, subdir=False, readonly=False, map_async = True, lock=True, readahead=False, meminit=False)
            with cop.begin(write = True) as txn_new:
                print("\nCreating Split #" + str(splitInd + 1))
                for idx in range(ensembleKeySplitCounts[splitInd]):
                    idx_encoded = str(idx).encode('ascii')
                    txn_new.put(idx_encoded, txn_old.get(key=idx_encoded))
            # Now that the changes are committed with closing of the with statement, check that entries are added.
            print("Split #" + str(splitInd + 1) + " completed. Description:")
            print(str(cop.stat()) + "\n")
            cop.close()
print("\nLMDB splitting completed. Closing original LMDB.")
env.close() (edited) 