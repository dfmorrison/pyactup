PyACTUp version 1.0.4
*********************

.. toctree::
   :maxdepth: 3
   :caption: Contents:


Introduction
============

PyACTUp is a lightweight Python implementation of a subset of the ACT-R  [#f1]_ cognitive architecture’s Declarative Memory,
suitable for incorporating into other Python models and applications. It is inspired by the ACT-UP [#f2]_ cognitive
modeling toolbox.

Typically PyACTUp is used by creating an experimental framework, or connecting to an existing experiment,
in the Python programming language, using one or more PyACTUp :class:`Memory` objects. The framework
or experiment asks these Memory objects to add chunks to themselves, describing things learned, and
retrieves these chunks or values derived from them at later times. A chunk, a learned item, contains one or more
slots or attributes, describing what is learned. Retrievals are driven by matching on the values of these
attributes. Each Memory object also has a notion of time, a non-negative, real number that is advanced as
the Memory object is used. Time in PyACTUp is a dimensionless quantity whose interpretation depends upon
the model or system in which PyACTUp is being used.
There are also several parameters
controlling these retrievals that can be configured in a Memory object, and detailed information can
be extracted from it describing the process it uses in making these retrievals.
The frameworks or experiments may be strictly algorithmic, may interact with human
subjects, or may be embedded in web sites.

PyACTUp is a library, or module, of `Python <http://www.python.org/>`_ code,  useful for creating
Python programs; it is not a stand alone application.
Some knowledge of Python programming is essential for using it.

PyACTUp is an ongoing project, and implements only a subset of ACT-R's Declarative Memory.
As it evolves it is likely that more of the ACT-R architecture will be incorporated into
it. Such future additions may also change some of the APIs exposed today, and some work
may be required to upgrade projects using the current version of PyACTUp to a later version.

.. [#f1] J. R. Anderson, D. Bothell, M. D. Byrne, S. Douglass,
         C. Lebiere, & Y. Qin (2004). An integrated theory of
         the mind. *Psychological Review* *111*, (4). 1036-1060.

.. [#f2] D. Reitter and C. Lebiere. (2010). Accountable Modeling in ACT-UP, a Scalable,
         Rapid-Prototyping ACT-R Implementation. In *Proceedings of the 10th International
         Conference on Cognitive Modeling (ICCM)*.



Installing PyACTUp
==================

PyACTUp requires Python version 3.6 or later. Recent versions of Mac OS X and recent Linux distributions
are likely to have a suitable version of Python pre-installed, but it may need to be invoked as ``python3``
instead of just ``python``, which latter often runs a 2.x version of Python instead. Use of a virtual environment,
which is recommended, often obviates the need for the ``python3``/``python`` distinction.
If it is not already installed, Python, for Windows, Mac OS X, Linux, or other Unices, can be
`downloaded from python.org <http://www.python.org/download/>`_, for free.

PyACTUp also works in recent versions of `PyPy <https://pypy.org/>`_, an alternative implementation to the usual CPython.
PyPy uses a just-in-time (JIT) compiler, which is a good match for PyACTUp, and PyACTUp models often
run five times faster in PyPy compared to CPython.

Note that PyACTUp is simply a Python module, a library, that is run as part of a larger
Python program. To build and run models using PyACTUp you do need to do
some Python programming. If you're new to Python, a good place to
start learning it is `The Python Tutorial <https://docs.python.org/3.6/tutorial/>`_.
To write and run a Python program you need to create and edit Python
source files, and then run them. If you are comfortable using the command
line, you can simply create and edit the files in your favorite text editor,
and run them from the command line. Many folks, though, are happier using
a graphical Integrated Development Environment (IDE).
`Many Python IDEs are available <https://wiki.python.org/moin/IntegratedDevelopmentEnvironments>`_.
One  is
`IDLE <https://docs.python.org/3.6/library/idle.html>`_,
which comes packaged with Python itself, so if you installed Python
you should have it available.

Normally, assuming you are connected to the internet, to install PyACTUp you should simply have to type at the command line

  .. parsed-literal:: pip install pyactup

Depending upon various possible variations in how Python and your machine are configured
you may have to modify the above in various ways

* you may need to ensure your virtual environment is activated

* you may need use an alternative scheme your Python IDE supports

* you may need to call it ``pip3`` instead of simply ``pip``

* you may need to precede the call to ``pip`` by ``sudo``

* you may need to use some combination of the above

If you are unable to install PyACTUp as above, you can instead
`download a tarball <https://bitbucket.org/dfmorrison/pyactup/downloads/?tab=downloads>`_.
The tarball will have a filename something like pyactup-1.0.tar.gz.
Assuming this file is at ``/some/directory/pyactup-1.0.tar.gz`` install it by typing at the command line

  .. parsed-literal:: pip install /some/directory/pyactup-1.0.tar.gz

Alternatively you can untar the tarball with

  .. parsed-literal:: tar -xf /some/directory/pyactup-1.0.tar.gz

and then change to the resulting directory and type

  .. parsed-literal:: python setup.py install


Background
==========

Activation
----------

A fundamental part of retrieving a chunk from a :class:`Memory` object is computing the activation of that chunk,
a real number describing
how likely it is to be recalled, based on how frequently and recently it has been added to the Memory, and how well it
matches the specifications of what is to be retrieved.

The activation, :math:`A_{i}` of chunk *i* is a sum of three
components,

  .. math:: A_{i} = B_{i} + \epsilon_{i} + P_{i}

the base level activation, the activation noise, and the partial matching correction.

Base level activation
~~~~~~~~~~~~~~~~~~~~~

The base level activation, :math:`B_{i}`, describes the frequency and recency of the chunk *i*,
and depends upon the ``decay`` parameter of Memory, *d*. In the normal case, when the
Memory's ``optimized_learning`` parameter is ``False``, the base level activation is computed using
the amount of time that has elapsed since each of the past appearances of *i*, which in the following
are denoted as the various :math:`t_{ij}`.


  .. math:: B_{i} = \ln(\sum_{j} t_{ij}^{-d})

If the Memory's ``optimized_learning`` parameter is ``True`` an approximation is used instead, often less taxing of
computational resources. It is particularly useful if the same chunks are expected to be seen many times, and assumes
that repeated experiences of the various chunks are distributed roughly evenly over time.
Instead of using the times of all the past occurrences of *i*, it uses *L*, the amount of time since
the first appearance of *i*, and *n*, a count of the number of times *i* has appeared.

  .. math:: B_{i} = \ln(\frac{n}{1 - d}) - d \ln(L)

Note that setting the ``decay`` parameter to ``None`` disables the computation of base level
activation. That is, the base level component of the total activation is zero in this case.

Activation noise
~~~~~~~~~~~~~~~~

The activation noise, :math:`\epsilon_{i}`, implements the stochasticity of retrievals from Memory.
It is sampled from a logistic distribution centered on zero. A Memory object has a scale
parameter, ``noise``, for this distribution. It is resampled each time the activation is computed.

Note that setting the ``noise`` parameter to zero results in supplying
no noise to the activation. This does not quite make operation of
PyACTUp deterministic, since retrievals of chunks with the same
activations are resolved randomly.

Partial Matching
~~~~~~~~~~~~~~~~

If the Memory’s ``mismatch`` parameter is ``None``, the partial matching correction, :math:`P_{i}`, is zero.
Setting the parameter to ``None`` is equivalent to setting it to :math:`-\inf`, ensuring that only chunks
that exactly match the retrival specification are considered.
Otherwise :math:`P_{i}` depends upon the similarities of the attributes of the chunk to those attributes
being sought in the retrieval and the value of the ``mismatch`` parameter.

PyACTUp normally uses a "natural" representation of similarities, where two values being completely similar, identical,
has a value of one; and being completely dissimilar has a value of zero; with various other degrees of similarity being
positive, real numbers less than one. Traditionally ACT-R instead uses a range of
similarities with the most dissimilar being a negative number, usually -1, and completely similar being zero.
If preferred, PyACTUp can be configured to use these ACT-R-style similarities by calling the
function ``use_actr_similarity`` with an argument of ``True``, resulting in the computations below being appropriately offset.

The ``set_similarity_function`` defines how to compute the similarity of values for a particular attribute.

.. 
   function ``set_similarity_function`` or the Memory method ``set_similarity``. If neither has been
   used in a way applicable to the two chunks being compared their similarity is one if they are the same
   chunk, and otherwise zero.

If the ``mismatch`` parameter has real value :math:`\mu` and the similarity of slot *k* of *i* to the desired
value of that slot in the retrieval is :math:`S_{ik}`, the partial matching correction is

  .. math:: P_{i} = \mu \sum_{k} (S_{ik} - 1)

The value of :math:`\mu` is normally positive, so :math:`P_{i}` is normally negative, and increasing dissimilarities
reduce the total activation, scaled by the value of :math:`\mu`.


Blending
--------

Besides retrieving an existing chunk, it is possible to retrieve an attribute value not present in any instance, a weighted
average, or blend, of the corresponding attribute values present in a set of existing chunks meeting some criteria.
Currently only real valued attributes can be blended.

A parameter, the ``temperature``, or :math:`\tau`, is used in constructing the blended value.
In PyACTUp the value of this parameter is by default the ``noise`` parameter used for activation noise,
multiplied by :math:`\sqrt{2}`. However it can be set independently of the ``noise``, if preferred.

If *m* is the set of chunks matching the criteria, and, for :math:`i \in m`, the activation
of chunk *i* is :math:`a_{i}`, we define a weight, :math:`w_{i}`, for the contribution *i*
makes to the blended value

  .. math:: w_{i} = e^{a_{i} / \tau}

If :math:`s_{i}` is the value of the slot or attribute of chunk *i* to be blended over,
the  blended value, *BV*, is then

  .. math:: BV =\, \sum_{i \in m}{\, \frac{w_{i}}{\sum_{j \in m}{w_{j}}} \; s_{i}}



API Reference
=============

.. automodule:: pyactup

.. autoclass:: Memory

   .. automethod:: learn

   .. automethod:: retrieve

   .. automethod:: blend

   .. automethod:: reset

   .. autoattribute:: time

   .. automethod:: advance

   .. autoattribute:: noise

   .. autoattribute:: decay

   .. autoattribute:: temperature

   .. autoattribute:: mismatch

   .. autoattribute:: optimized_learning

   .. autoattribute:: activation_history

   .. automethod:: forget

.. autofunction:: set_similarity_function

.. autofunction:: use_actr_similarity


Examples
========

Rock, paper, scissors
---------------------

This is an example of using PyACTUp to model the
`Rock, Paper, Scissors game <https://www.wrpsa.com/the-official-rules-of-rock-paper-scissors/>`_.
Both players are modeled, and attempt to chose their moves based on their expectations of
the move that will be made by their opponents. The two players differ in how much of the
prior history they consider in creating their expectations.

Download the source code for the `rock, paper, scissors <http://halle.psy.cmu.edu/pyactup/examples/rps.py>`_ example.

.. code-block:: python
    :linenos:

    # Rock, paper, scissors example using pyactup

    import pyactup
    import random

    DEFAULT_ROUNDS = 100
    MOVES = ["paper", "rock", "scissors"]
    N_MOVES = len(MOVES)

    m = pyactup.Memory(noise=0.1)

    def defeat_expectation(**kwargs):
        # Generate expectation matching supplied conditions and play the move that defeats.
        # If no expectation can be generate, chooses a move randomly.
        expectation = (m.retrieve(**kwargs) or {}).get("move")
        if expectation:
            return MOVES[(MOVES.index(expectation) - 1) % N_MOVES]
        else:
            return random.choice(MOVES)

    def safe_element(list, i):
        try:
            return list[i]
        except IndexError:
            return None

    def main(rounds=DEFAULT_ROUNDS):
        # Plays multiple rounds of r/p/s of a lag 1 player (player1) versus a
        # lag 2 player (player2).
        plays1 = []
        plays2 = []
        score = 0
        for r in range(rounds):
            move1 = defeat_expectation(player="player2",
                                       ultimate=safe_element(plays2, -1))
            move2 = defeat_expectation(player="player1",
                                       ultimate=safe_element(plays1, -1),
                                       penultimate=safe_element(plays1, -2))
            winner = (MOVES.index(move2) - MOVES.index(move1) + N_MOVES) % N_MOVES
            score += -1 if winner == 2 else winner
            print("Round {:3d}\tPlayer 1: {:8s}\tPlayer 2: {:8s}\tWinner: {}\tScore: {:4d}".format(
                r, move1, move2, winner, score))
            m.learn(player="player1",
                    ultimate=safe_element(plays1, -1),
                    penultimate=safe_element(plays1, -2),
                    move=move1)
            m.learn(player="player2", ultimate=safe_element(plays2, -1), move=move2)
            plays1.append(move1)
            plays2.append(move2)
            m.advance()


    if __name__ == '__main__':
        main()

Here's the result of running it once. Because the model is stochastic, if you run it yourself
the results will be different.

.. code-block::

    $ python rps.py
    Round   0   Player 1: rock          Player 2: scissors      Winner: 1       Score:    1
    Round   1   Player 1: rock          Player 2: scissors      Winner: 1       Score:    2
    Round   2   Player 1: rock          Player 2: rock          Winner: 0       Score:    2
    Round   3   Player 1: scissors      Player 2: paper         Winner: 1       Score:    3
    Round   4   Player 1: rock          Player 2: scissors      Winner: 1       Score:    4
    Round   5   Player 1: paper         Player 2: paper         Winner: 0       Score:    4
    Round   6   Player 1: rock          Player 2: scissors      Winner: 1       Score:    5
    Round   7   Player 1: scissors      Player 2: paper         Winner: 1       Score:    6
    Round   8   Player 1: rock          Player 2: paper         Winner: 2       Score:    5
    Round   9   Player 1: rock          Player 2: scissors      Winner: 1       Score:    6
    Round  10   Player 1: scissors      Player 2: rock          Winner: 2       Score:    5
    Round  11   Player 1: scissors      Player 2: paper         Winner: 1       Score:    6
    Round  12   Player 1: rock          Player 2: scissors      Winner: 1       Score:    7
    Round  13   Player 1: paper         Player 2: paper         Winner: 0       Score:    7
    Round  14   Player 1: rock          Player 2: paper         Winner: 2       Score:    6
    Round  15   Player 1: scissors      Player 2: rock          Winner: 2       Score:    5
    Round  16   Player 1: scissors      Player 2: paper         Winner: 1       Score:    6
    Round  17   Player 1: rock          Player 2: paper         Winner: 2       Score:    5
    Round  18   Player 1: scissors      Player 2: scissors      Winner: 0       Score:    5
    Round  19   Player 1: scissors      Player 2: rock          Winner: 2       Score:    4
    Round  20   Player 1: scissors      Player 2: paper         Winner: 1       Score:    5
    Round  21   Player 1: rock          Player 2: paper         Winner: 2       Score:    4
    Round  22   Player 1: scissors      Player 2: scissors      Winner: 0       Score:    4
    Round  23   Player 1: paper         Player 2: rock          Winner: 1       Score:    5
    Round  24   Player 1: scissors      Player 2: rock          Winner: 2       Score:    4
    Round  25   Player 1: scissors      Player 2: scissors      Winner: 0       Score:    4
    Round  26   Player 1: paper         Player 2: paper         Winner: 0       Score:    4
    Round  27   Player 1: scissors      Player 2: rock          Winner: 2       Score:    3
    Round  28   Player 1: scissors      Player 2: rock          Winner: 2       Score:    2
    Round  29   Player 1: paper         Player 2: paper         Winner: 0       Score:    2
    Round  30   Player 1: rock          Player 2: rock          Winner: 0       Score:    2
    Round  31   Player 1: scissors      Player 2: rock          Winner: 2       Score:    1
    Round  32   Player 1: paper         Player 2: rock          Winner: 1       Score:    2
    Round  33   Player 1: paper         Player 2: rock          Winner: 1       Score:    3
    Round  34   Player 1: paper         Player 2: rock          Winner: 1       Score:    4
    Round  35   Player 1: paper         Player 2: scissors      Winner: 2       Score:    3
    Round  36   Player 1: scissors      Player 2: scissors      Winner: 0       Score:    3
    Round  37   Player 1: rock          Player 2: rock          Winner: 0       Score:    3
    Round  38   Player 1: paper         Player 2: rock          Winner: 1       Score:    4
    Round  39   Player 1: paper         Player 2: paper         Winner: 0       Score:    4
    Round  40   Player 1: rock          Player 2: scissors      Winner: 1       Score:    5
    Round  41   Player 1: paper         Player 2: rock          Winner: 1       Score:    6
    Round  42   Player 1: paper         Player 2: scissors      Winner: 2       Score:    5
    Round  43   Player 1: paper         Player 2: scissors      Winner: 2       Score:    4
    Round  44   Player 1: rock          Player 2: scissors      Winner: 1       Score:    5
    Round  45   Player 1: rock          Player 2: scissors      Winner: 1       Score:    6
    Round  46   Player 1: rock          Player 2: rock          Winner: 0       Score:    6
    Round  47   Player 1: paper         Player 2: paper         Winner: 0       Score:    6
    Round  48   Player 1: rock          Player 2: scissors      Winner: 1       Score:    7
    Round  49   Player 1: paper         Player 2: rock          Winner: 1       Score:    8
    Round  50   Player 1: scissors      Player 2: paper         Winner: 1       Score:    9
    Round  51   Player 1: rock          Player 2: rock          Winner: 0       Score:    9
    Round  52   Player 1: scissors      Player 2: scissors      Winner: 0       Score:    9
    Round  53   Player 1: paper         Player 2: rock          Winner: 1       Score:   10
    Round  54   Player 1: scissors      Player 2: rock          Winner: 2       Score:    9
    Round  55   Player 1: paper         Player 2: paper         Winner: 0       Score:    9
    Round  56   Player 1: rock          Player 2: rock          Winner: 0       Score:    9
    Round  57   Player 1: scissors      Player 2: scissors      Winner: 0       Score:    9
    Round  58   Player 1: paper         Player 2: scissors      Winner: 2       Score:    8
    Round  59   Player 1: rock          Player 2: rock          Winner: 0       Score:    8
    Round  60   Player 1: scissors      Player 2: rock          Winner: 2       Score:    7
    Round  61   Player 1: scissors      Player 2: scissors      Winner: 0       Score:    7
    Round  62   Player 1: paper         Player 2: paper         Winner: 0       Score:    7
    Round  63   Player 1: rock          Player 2: paper         Winner: 2       Score:    6
    Round  64   Player 1: scissors      Player 2: rock          Winner: 2       Score:    5
    Round  65   Player 1: paper         Player 2: rock          Winner: 1       Score:    6
    Round  66   Player 1: paper         Player 2: paper         Winner: 0       Score:    6
    Round  67   Player 1: paper         Player 2: scissors      Winner: 2       Score:    5
    Round  68   Player 1: paper         Player 2: scissors      Winner: 2       Score:    4
    Round  69   Player 1: rock          Player 2: scissors      Winner: 1       Score:    5
    Round  70   Player 1: rock          Player 2: rock          Winner: 0       Score:    5
    Round  71   Player 1: scissors      Player 2: paper         Winner: 1       Score:    6
    Round  72   Player 1: rock          Player 2: scissors      Winner: 1       Score:    7
    Round  73   Player 1: rock          Player 2: rock          Winner: 0       Score:    7
    Round  74   Player 1: scissors      Player 2: rock          Winner: 2       Score:    6
    Round  75   Player 1: scissors      Player 2: scissors      Winner: 0       Score:    6
    Round  76   Player 1: paper         Player 2: scissors      Winner: 2       Score:    5
    Round  77   Player 1: paper         Player 2: paper         Winner: 0       Score:    5
    Round  78   Player 1: rock          Player 2: scissors      Winner: 1       Score:    6
    Round  79   Player 1: paper         Player 2: rock          Winner: 1       Score:    7
    Round  80   Player 1: scissors      Player 2: paper         Winner: 1       Score:    8
    Round  81   Player 1: rock          Player 2: paper         Winner: 2       Score:    7
    Round  82   Player 1: rock          Player 2: rock          Winner: 0       Score:    7
    Round  83   Player 1: scissors      Player 2: rock          Winner: 2       Score:    6
    Round  84   Player 1: paper         Player 2: rock          Winner: 1       Score:    7
    Round  85   Player 1: paper         Player 2: paper         Winner: 0       Score:    7
    Round  86   Player 1: rock          Player 2: scissors      Winner: 1       Score:    8
    Round  87   Player 1: paper         Player 2: rock          Winner: 1       Score:    9
    Round  88   Player 1: scissors      Player 2: rock          Winner: 2       Score:    8
    Round  89   Player 1: paper         Player 2: paper         Winner: 0       Score:    8
    Round  90   Player 1: rock          Player 2: scissors      Winner: 1       Score:    9
    Round  91   Player 1: paper         Player 2: scissors      Winner: 2       Score:    8
    Round  92   Player 1: paper         Player 2: rock          Winner: 1       Score:    9
    Round  93   Player 1: scissors      Player 2: paper         Winner: 1       Score:   10
    Round  94   Player 1: rock          Player 2: scissors      Winner: 1       Score:   11
    Round  95   Player 1: rock          Player 2: paper         Winner: 2       Score:   10
    Round  96   Player 1: rock          Player 2: rock          Winner: 0       Score:   10
    Round  97   Player 1: scissors      Player 2: paper         Winner: 1       Score:   11
    Round  98   Player 1: rock          Player 2: scissors      Winner: 1       Score:   12
    Round  99   Player 1: paper         Player 2: paper         Winner: 0       Score:   12



Safe, risky
-----------

This is an example of using PyACTUp to create an Instance Based Learning (IBL) [#f3]_ model of
a binary choice task, exhibiting risk aversion. A choice is made between two options, one safe
and the other risky. The safe choice always pays out one unit. The risky choice is random, paying
out three units one third of the time and zero units the rest. In this example code the choice
is made by each virtual participant over the course of 60 rounds, learning from the experience
of previous rounds. And the results are collected over 10,000 independent participants, and
the number of risky choices, averaged over all participants, is plotted.

This code uses two other Python packages, `matplotlib <https://matplotlib.org/>`_
and `tqdm <https://tqdm.github.io/>`_.
Neither is actually used by the model proper, and the code can be rearranged to dispense with
them, if preferred. ``Matplotlib`` is used to generate a graph of the results, and ``tqdm``
to display a progress indicator, as this example takes on the order of twenty seconds to run
in CPython.

Download the source code for the `safe, risky <http://halle.psy.cmu.edu/pyactup/examples/safe_risky.py>`_ example.

.. code-block:: python
    :linenos:

    import pyactup
    import random

    import matplotlib.pyplot as plt

    from tqdm import tqdm

    PARTICIPANTS = 10_000
    ROUNDS = 60

    risky_chosen = [0] * ROUNDS
    m = pyactup.Memory()
    for p in tqdm(range(PARTICIPANTS)):
        m.reset()
        # prepopulate some instances to ensure initial exploration
        m.learn(choice="safe", outcome=1)
        m.learn(choice="risky", outcome=0)
        m.learn(choice="risky", outcome=2)
        m.advance()
        for r in range(ROUNDS):
            safe_bv = m.blend("outcome", choice="safe")
            risky_bv = m.blend("outcome", choice="risky")
            if risky_bv > safe_bv or (risky_bv == safe_bv and random.random() < 0.5):
                choice = "risky"
                payoff = 3 if random.random() < 1/3 else 0
                risky_chosen[r] += 1
            else:
                choice = "safe"
                payoff = 1
            m.learn(choice=choice, outcome=payoff)
            m.advance()

    plt.plot(range(ROUNDS), [ v / PARTICIPANTS for v in risky_chosen])
    plt.ylim([0, 1])
    plt.ylabel("fraction choosing risky")
    plt.xlabel("round")
    plt.title(f"Safe (1 always) versus risky (3 × ⅓, 0 × ⅔)\nσ={m.noise}, d={m.decay}")
    plt.show()

The result of running this is

    .. image:: safe_risky_graph.png



.. [#f3] Cleotilde Gonzalez, Javier F. Lerch and Christian Lebiere (2003),
         `Instance-based learning in dynamic decision making,
         <http://www.sciencedirect.com/science/article/pii/S0364021303000314>`_
         *Cognitive Science*, *27*, 591-635. DOI: 10.1016/S0364-0213(03)00031-4.
