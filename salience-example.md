### A Simple Example of Salience for ACT-UP

> We think in generalities, but we live in detail. – Alfred North Whitehead

This is a simple example of salience computation, just to make sure I understand how it works before adding
it to PyACTUp. First we create a small ensemble of chunks, each with three slots, $r$, $h$ and $v$, each holding
a real number. While we have six experiences, there are only four chunks, because of repetition at different times.

    m = pyactup.Memory(temperature=1, noise=0, mismatch=1)

    m.learn({"r": 1, "h": 1, "v": 1})
    m.advance()
    m.learn({"r": 3, "h": 3, "v": 27})
    m.advance()
    m.learn({"r": 1, "h": 3, "v": 3})
    m.advance()
    m.learn({"r": 1, "h": 1, "v": 1})
    m.advance()
    m.learn({"r": 1, "h": 1, "v": 1})
    m.advance()
    m.learn({"r": 3, "h": 1, "v": 9})
    m.advance()

    om.print_chunks()
    +------------+-------------------------+------------------+-----------------------+------------------+
    | chunk name |      chunk contents     | chunk created at | chunk reference count | chunk references |
    +------------+-------------------------+------------------+-----------------------+------------------+
    |    0000    |  'h': 1, 'r': 1, 'v': 1 |        0         |           3           |     0, 3, 4      |
    |    0001    | 'h': 3, 'r': 3, 'v': 27 |        1         |           1           |        1         |
    |    0002    |  'h': 3, 'r': 1, 'v': 3 |        2         |           1           |        2         |
    |    0003    |  'h': 1, 'r': 3, 'v': 9 |        5         |           1           |        5         |
    +------------+-------------------------+------------------+-----------------------+------------------+

We will compute a blended value for
