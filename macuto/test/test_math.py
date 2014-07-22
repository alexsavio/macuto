__author__ = 'alexandre'


from .. import math

def test_takespread():
    """
    @param sequence:
    @param num:
    @return:
    """
    #length = float(len(sequence))
    #for i in range(num):
    #    yield sequence[int(np.ceil(i * length / num))]
    result = [0, 4, 7]
    for idx, val in enumerate(math.takespread(range(10), 3)):
        assert(val == result[idx])


def test_makespread():
    """
    @param sequence:
    @param num:
    @return:
    """
    #length = float(len(sequence))
    #seq = np.array(sequence)
    #return seq[np.ceil(np.arange(num) * length / num).astype(int)]
    assert(all(math.makespread(range(10), 3) == [0, 4, 7]))
