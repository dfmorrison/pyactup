PyACTUp version 2.0.12
**********************

.. toctree::
   :maxdepth: 3
   :caption: Contents:

.. note::
    The argument signatures of several common methods were changed in version 2.0. Existing models
    written for earlier versions of PyACTUp will need to be updated, typically by making relatively
    simple syntactic changes to the relevant calls. See :ref:`upgrading` for details.



Introduction
============

PyACTUp is a lightweight Python implementation of a subset of the ACT-R  [#f1]_ cognitive architecture’s Declarative Memory,
suitable for incorporating into other Python models and applications. Its creation was inspired by the ACT-UP [#f2]_ cognitive
modeling toolbox.

Typically PyACTUp is used by creating an experimental framework, or connecting to an existing experiment,
in the Python programming language, using one or more PyACTUp :class:`Memory` objects. The framework
or experiment asks these Memory objects to add chunks to themselves, describing things learned, and
retrieves these chunks or values derived from them at later times. A chunk, a learned item, contains one or more
slots or attributes, describing what is learned. Retrievals are driven by matching on the values of these
attributes. Each Memory object also has a notion of time, a non-negative, real number that is advanced as
the Memory object is used. Time in PyACTUp is a dimensionless quantity whose interpretation depends upon
the model or system in which PyACTUp is being used. Note that in most cases the likelihood of retrievals does not depend
upon the actual scale of time used, only on the ratios of the various values, but subtle dependencies
can ensue when partial matching or other perturbations involving the activation rather than the probability
of retrieve are used.
There are also several parameters
controlling these retrievals that can be configured in a Memory object, and detailed information can
be extracted from it describing the process it uses in making these retrievals.
The frameworks or experiments may be strictly algorithmic, may interact with human
subjects, or may be embedded in web sites.

PyACTUp is a library, or module, of `Python <http://www.python.org/>`_ code,  useful for creating
Python programs; it is not a stand alone application.
Some knowledge of Python programming is essential for using it.

PyACTUp is an ongoing project, and implements only a subset of ACT-R's Declarative Memory.
As it evolves it is possible that more of the ACT-R architecture will be incorporated into
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

The latest version of PyACTUp can be downloaded and installed from PyPi with  ``pip``:

  .. parsed-literal:: pip install pyactup

Use of a virtual environment for Python, such as `venv <https://docs.python.org/3.8/library/venv.html>`_
or `Anaconda <https://www.anaconda.com/>`_ is recommended.

PyACTUp requires Python version 3.8 or later.

Note that PyACTUp is simply a Python module, a library, that is run as part of a larger
Python program. To build and run models using PyACTUp you do need to do
some Python programming. If you're new to Python, a good place to
start learning it is `The Python Tutorial <https://docs.python.org/3.8/tutorial/>`_.
To write and run a Python program you need to create and edit Python
source files, and then run them. If you are comfortable using the command
line, you can simply create and edit the files in your favorite text editor,
and run them from the command line. Many folks, though, are happier using
a graphical Integrated Development Environment (IDE).
`Many Python IDEs are available <https://wiki.python.org/moin/IntegratedDevelopmentEnvironments>`_.
One  is
`IDLE <https://docs.python.org/3.8/library/idle.html>`_,
which comes packaged with Python itself, so if you installed Python
you should have it available.

The PyACTUp sources are available
at `https://github.org/dfmorrison/pyactup/ <https://github.org/dfmorrison/pyactup/>`_.
This document is also available
at `http://koalemos.psy.cmu.edu/pyactup/ <http://koalemos.psy.cmu.edu/pyactup/>`_.

Mailing List
============

There is a `mailing list <https://lists.andrew.cmu.edu/mailman/listinfo/pyactup-users>`_ for those interested in PyACTUp and its development.


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

the base-level activation, the activation noise, and the partial matching correction.

Base-level activation
~~~~~~~~~~~~~~~~~~~~~

The base-level activation, :math:`B_{i}`, describes the frequency and recency of the chunk *i*,
and depends upon the ``decay`` parameter of Memory, *d*. In the normal case, when the
Memory's ``optimized_learning`` parameter is ``False``, the base-level activation is computed using
the amount of time that has elapsed since each of the past appearances of *i*, which in the following
are denoted as the various :math:`t_{ij}`.


  .. math:: B_{i} = \ln(\sum_{j} t_{ij}^{-d})

If the Memory's ``optimized_learning`` parameter is ``True`` an approximation is used instead, sometimes less taxing of
computational resources. It is particularly useful if the same chunks are expected to be seen many times, and assumes
that repeated experiences of the various chunks are distributed roughly evenly over time.
Instead of using the times of all the past occurrences of *i*, it uses *L*, the amount of time since
the first appearance of *i*, and *n*, a count of the number of times *i* has appeared.

  .. math:: B_{i} = \ln(\frac{n}{1 - d}) - d \ln(L)

The ``optimized_learning`` parameter may also be set to a positive integer. This specifies a number of most recent
reinforcements of a chunk to be used to compute the base-level activation in the normal way, with the contributions
of any older than those approximated using a formula similar to the preceding.

Note that setting the ``decay`` parameter to ``None`` disables the computation of base-level
activation. That is, the base-level component of the total activation is zero in this case.
This is different than setting the ``decay`` parameter to zero which still computes the
base level activation component, albeit using only frequency with no decay over time.

Activation noise
~~~~~~~~~~~~~~~~

The activation noise, :math:`\epsilon_{i}`, implements the stochasticity of retrievals from Memory.
It is sampled from a logistic distribution centered on zero. A Memory object has a scale
parameter, ``noise``, for this distribution. It is normally resampled each time the activation is computed.

For some esoteric purposes when a chunk’s activation is computed repeatedly at the same time it
may be desired to have all these same-time activations of a chunk use the same sample of activation noise.
While this is rarely needed, and best avoided unless absolutely necessary, when it is needed the ``fixed_noise``
context manager can be used.

Note that setting the ``noise`` parameter to zero results in supplying
no noise to the activation. This does not quite make operation of
PyACTUp deterministic, since retrievals of chunks with the same
activations are resolved randomly.

Partial Matching
~~~~~~~~~~~~~~~~

If the Memory’s ``mismatch`` parameter is ``None``, the partial matching correction, :math:`P_{i}`, is zero.
Setting the parameter to ``None`` is equivalent to setting it to ∞, ensuring that only chunks
that exactly match the retrival specification are considered.
Otherwise :math:`P_{i}` depends upon the similarities of the attributes of the chunk to those attributes
being sought in the retrieval and the value of the ``mismatch`` parameter.
When considering chunks in partial retrievals or blending operations attributes for which no similarity
function has been defined are treated as exact matches; chunks not matching these attributes are not
included in the partial retrieval or blending operation.

PyACTUp normally uses a "natural" representation of similarities, where two values being completely similar, typically
Python ``==``,
has a value of one; and being completely dissimilar has a value of zero; with various other degrees of similarity being
positive, real numbers less than one. Traditionally ACT-R instead uses a range of
similarities with the most dissimilar being a negative number, usually -1, and completely similar being zero.
If preferred, PyACTUp can be configured to use these ACT-R-style similarities by setting
the ``use_actr_similarity`` attribute of a ``Memory`` object to ``True``, resulting in the computations
below being appropriately offset.

The ``similarity`` method defines how to compute the similarity of values for a particular attribute when
it appears in a ``Memory``’s chunks. A function is supplied to this method to be applied to values of the
attributes of given names, this function returning a similarity value. In addition, the ``similarity`` method
can assign a weight, :math:`\omega`, to these slots, allowing the mismatch contributions of multiple slots
to be scaled with respect to one another. If not explicitly supplied this weight defaults to one.o

If the ``mismatch`` parameter has real value :math:`\mu`, the similarity of slot *k* of *i* to the desired
value of that slot in the retrieval is :math:`S_{ik}`, and the similarity weight of slot *k* is :math:`\omega_{k}`,
the partial matching correction is

  .. math:: P_{i} = \mu \sum_{k} \omega_{k} (S_{ik} - 1)

The value of :math:`\mu` is normally positive, so :math:`P_{i}` is normally negative, and increasing dissimilarities
reduce the total activation, scaled by the value of :math:`\mu`.


Blending
--------

Besides retrieving an existing chunk, it is possible to retrieve an attribute value not present in any instance, a weighted
average, or blend, of the corresponding attribute values present in a set of existing chunks meeting some criteria.
Typically only real valued attributes are blended.

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

It is also possible to perform a discrete blending operation where an exisiting slot value is
returned, albeit one possibly not appearing an any chunk that directly matches the criteria,
it instead resulting from a blending operation using the same weights as above.



API Reference
=============

.. automodule:: pyactup

.. autoclass:: Memory

   .. automethod:: learn

   .. automethod:: retrieve

   .. automethod:: blend

   .. automethod:: best_blend

   .. automethod:: discrete_blend

   .. automethod:: reset

   .. autoattribute:: time

   .. automethod:: advance

   .. autoattribute:: noise

   .. autoattribute:: decay

   .. autoattribute:: temperature

   .. autoattribute:: mismatch

   .. autoattribute:: threshold

   .. autoattribute:: optimized_learning

   .. automethod:: similarity

   .. autoattribute:: chunks

   .. automethod:: print_chunks

   .. autoattribute:: activation_history

   .. automethod:: forget

   .. autoattribute:: current_time

   .. autoattribute:: fixed_noise

   .. autoattribute:: use_actr_similarity



Examples
========

Rock, paper, scissors
---------------------

This is an example of using PyACTUp to model the
`Rock, Paper, Scissors game <https://www.wrpsa.com/the-official-rules-of-rock-paper-scissors/>`_.
Both players are modeled, and attempt to chose their moves based on their expectations of
the move that will be made by their opponents. The two players differ in how much of the
prior history they consider in creating their expectations.

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
        expectation = (m.retrieve(kwargs) or {}).get("move")
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
            m.learn({"player": "player1",
                     "ultimate": safe_element(plays1, -1),
                     "penultimate": safe_element(plays1, -2),
                     "move": move1})
            m.learn({"player": "player2",
                     "ultimate": safe_element(plays2, -1),
                     "move": move2},
                    advance=2)
            plays1.append(move1)
            plays2.append(move2)


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
of previous rounds. The results are collected over 10,000 independent participants, and
the number of risky choices at each round, averaged over all participants, is plotted.

This code uses two other Python packages, `matplotlib <https://matplotlib.org/>`_
and `tqdm <https://tqdm.github.io/>`_.
Neither is actually used by the model proper, and the code can be rearranged to dispense with
them, if preferred. ``Matplotlib`` is used to draw a graph of the results, and ``tqdm``
to display a progress indicator, as this example takes on the order of a minute to run
in CPython.

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
        for c, o in (("safe", 1), ("risky", 0), ("risky", 2)):
            m.learn({"choice": c, "outcome": o})
        m.advance()
        for r in range(ROUNDS):
            choice, bv = m.best_blend("outcome", ("safe", "risky"), "choice")
            if choice == "risky":
                payoff = 3 if random.random() < 1/3 else 0
                risky_chosen[r] += 1
            else:
                payoff = 1
            m.learn({"choice": choice, "outcome": payoff})
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


Changes to PyACTUp
==================

Changes between versions 2.0.2 and 2.0.11
-----------------------------------------

* The index is now a tuple instead of a list.

Changes between versions 2.0.2 and 2.0.11
-----------------------------------------

* Adjusted copyrights and documentation.
* Fixed a bug when attempting to partially match with a mismatch parameter of zero.

Changes between versions 2.0 and 2.0.2
--------------------------------------

* The canonical home for PyACT is now on GitHub instead of Bitbucket.

.. _upgrading:

Changes between versions 1.1.4 and 2.0
--------------------------------------

* PyACTUp now requires Python 3.8 or later.
* Changed the arguments to learn(), forget(), retrieve() and blend().
* There is now a new discrete_blend() method.
* Similarity functions are now per-memory, and are set using the similarity() method,
  there no longer being a set_similarity_function() function.
* Similarities can now have weights, also set with the similarity() method.
* The optimized_learning parameter can now be set like other parameters, and there
  is no longer an optimized_learning parameter to the reset() method.
* The optimized_learning parameter can now take positive integers as its value,
  allowing a mixture of normal and approximate activation computations.
* There is no longer any auto-advancing before learn() and/or after retrieve() or blend();
  while auto-advancing slightly simplified trivial demonstration models, it invariably
  caused difficulties with real models, and sowed confusion. As an intermediate measure
  there is still an advance= keyword argument available in learn().
* Some operations have been significantly speeded up for models with many chunks
  and/or rehearsals of those chunks, particularly if the new index argument to Memory() is used.
* General tidying and minor bug fixes.

When upgrading existing 1.x models to version 2.0 or later some syntactic changes will nearly always have to be made,
in particular to calls to :meth:`learn`, :meth:`retrieve` and :meth:`blend`. What in 1.x were expressed (mis)using keyword
arguments are now passed as a single argument, typically a dictionary. In most cases this simply requires wrapping
curly braces around the relevant arguments, quote marks around the attribute/slot names, and replacing the equals
signs by colons. If real keyword arguments, such as ``advance=``, are used they should all be moved after the dictionary.
Note that it is now no longer necessary to avoid using Python keywords or possible keyword arguments of the various
methods as attribute names. And this change also allows sensible extension of the PyACTUp API by adding further keyword
arguments to the relevant methods.

If you are using partial matching you will also have to replace calls to :func:`set_similarity_function` by
the :meth:`similarity` method of Memory objects. This method also takes slightly different arguments than
the former function.

Also since auto-advance is no longer done, explict calls to advance() (or additions of advance= to learn())
may need to be added at appropriate points in older code.

For example, what in 1.x would have been expressed as

.. code-block:: python

    set_similarity_function(cubic_similarity, "weight", "volume")
    m.learn(color="red", size=4)
    x = m.retrieve(color="blue")
    bv = m.blend("size“, color="green")

might become in 2.0

.. code-block:: python

    m.similarity(["weight", "volume"], cubic_similarity)
    m.learn({"color": "red", "size": 4})
    m.advance()
    x = m.retrieve({"color": "blue"})
    bv = m.blend("size", {"color": "green"})


Changes between versions 1.1.3 and 1.1.4
----------------------------------------

* Changed the underlying representation of chunks and now use numpy to compute base activations, speeding up some models.
* Added chunks attribute and print_chunks() method.


Changes between versions 1.0.9 and 1.1.3
----------------------------------------

* Added current_time and fixed_noise conext managers.
* PyACTUp now requires Python version 3.7 or later.


Changes between versions 1.0 and 1.0.9
--------------------------------------

* Allow disabling base-level activation.
* Cache similarity computations.
* Add rehearse parameter to retrieve() method.
* Add best_blend() method.
* Add preserve_prepopulated argument to reset().
