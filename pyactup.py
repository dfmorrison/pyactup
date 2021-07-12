# Copyright (c) 2018-2021 Carnegie Mellon University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""PyACTUp is a lightweight Python implementation of a subset of the ACT-R cognitive
architecture’s Declarative Memory, suitable for incorporating into other Python models and
applications. It is inspired by the ACT-UP cognitive modeling toolbox.

Typically PyACTUp is used by creating an experimental framework, or connecting to an
existing experiment, in the Python programming language, using one or more PyACTUp
:class:`Memory` objects. The framework or experiment asks these Memory objects to add
chunks to themselves, describing things learned, and retrieves these chunks or values
derived from them at later times. A chunk, a learned item, contains one or more slots or
attributes, describing what is learned. Retrievals are driven by matching on the values of
these attributes. Each Memory object also has a notion of time, a non-negative, real
number that is advanced as the Memory object is used. Time in PyACTUp is a dimensionless
quantity whose interpretation depends upon the model or system in which PyACTUp is being
used. There are also several parameters controlling these retrievals that can be
configured in a Memory object, and detailed information can be extracted from it
describing the process it uses in making these retrievals. The frameworks or experiments
may be strictly algorithmic, may interact with human subjects, or may be embedded in web
sites.
"""

__version__ = "1.1.4"

if "dev" in __version__:
    print("PyACTUp version", __version__)

import collections.abc as abc
import csv
import io
import math
import numpy as np
import operator
import pylru
import random
import sys

from contextlib import contextmanager
from numbers import Number
from prettytable import PrettyTable
from warnings import warn

__all__ = ("Memory", "set_similarity_function", "use_actr_similarity")

DEFAULT_NOISE = 0.25
DEFAULT_DECAY = 0.5
DEFAULT_THRESHOLD = -10.0
DEFAULT_LEARNING_TIME_INCREMENT = 1
DEFAULT_RETRIEVAL_TIME_INCREMENT = 0

MINIMUM_TEMPERATURE = 0.01

LN_CACHE_SIZE = 1000
SIMILARITY_CACHE_SIZE = 10_000
NOISE_VALUES_SIZE = 1000
MAXIMUM_RANDOM_SEED = 2**62

class Memory(dict):
    """A cognitive entity containing a collection of learned things, its chunks.
    A Memory object also contains a current time, which can be queried as the :attr:`time`
    property.

    The number of distinct chunks a Memory contains can be determined with Python's
    usual :func:`len` function.

    A Memory has several parameters controlling its behavior: :attr:`noise`,
    :attr:`decay`, :attr:`temperature`, :attr:`threshold`, :attr:`mismatch`,
    :attr:`learning_time_increment`, attr:`retrieval_time_increment` and
    :attr:`optimized_learning`. All can be queried, and most set, as properties on the
    Memory object. When creating a Memory object their initial values can be supplied as
    parameters.

    A Memory object can be serialized with
    `pickle <https://docs.python.org/3.6/library/pickle.html>`_, allowing Memory objects
    to be saved to and restored from persistent storage.

    If, when creating a ``Memory`` object, any of *noise*, *decay* or *mismatch* are
    negative, or if *temperature* is less than 0.01, a :exc:`ValueError` is raised.

    """

    def __init__(self,
                 noise=DEFAULT_NOISE,
                 decay=DEFAULT_DECAY,
                 temperature=None,
                 threshold=DEFAULT_THRESHOLD,
                 mismatch=None,
                 learning_time_increment=DEFAULT_LEARNING_TIME_INCREMENT,
                 retrieval_time_increment=DEFAULT_RETRIEVAL_TIME_INCREMENT,
                 optimized_learning=False):
        self._temperature_param = 1 # will be reset below, but is needed for noise assignment
        self._activation_noise_cache = None
        self._activation_noise_cache_time = None
        self._noise = None
        self._decay = None
        self.noise = noise
        self._optimized_learning = False
        self.decay = decay
        if temperature is None and not self._validate_temperature(None, noise):
            warn(f"A noise of {noise} and temperature of None will make the temperature too low; setting temperature to 1")
            self.temperature = 1
        else:
            self.temperature = temperature
        self.threshold = threshold
        self.mismatch = mismatch
        self._learning_time_increment = learning_time_increment
        self._retrieval_time_increment = retrieval_time_increment
        self._activation_history = None
        # Initialize the noise RNG from the parent Python RNG, in case the latter gets seeded for determinancy.
        self._rng = np.random.default_rng([random.randint(0, MAXIMUM_RANDOM_SEED) for i in range(16)])
        self._noise_values = None
        self._next_noise_value = NOISE_VALUES_SIZE
        self.reset(optimized_learning=bool(optimized_learning))

    def __repr__(self):
        return f"<Memory {dict(self.items())}>"

    def __str__(self):
        return f"<Memory {id(self)}>"

    def reset(self, preserve_prepopulated=False, optimized_learning=None):
        """Deletes the Memory's chunks and resets its time to zero.
        If *preserve_prepopulated* is false it deletes all chunks; if it is true it
        deletes all chunk references later than time zero, completely deleting those
        chunks that were created at time greater than zero.
        If *optimized_learning* is not None it sets the Memory's :attr:`optimized_learning`
        parameter; otherwise it leaves it unchanged. This Memory's :attr:`noise`,
        :attr:`decay`, :attr:`temperature`, :attr:`threshold` and :attr:`mismatch`
        parameters are left unchanged.
        """
        if optimized_learning and self._decay >= 1:
            raise RuntimeError(f"Optimized learning cannot be enabled if the decay, {self._decay}, is not less than 1")
        if preserve_prepopulated:
            preserved = {k: v for k, v in self.items() if v._creation == 0}
        self.clear()
        self._time = 0
        if optimized_learning is not None:
            self._optimized_learning = bool(optimized_learning)
        if preserve_prepopulated:
            for k, v in preserved.items():
                v._references = np.empty(1, dtype=int) if self._optimized_learning else np.array([0])
                v._reference_count = 1
                v._base_activation_time = None
                v._base_activation = None
                self[k] = v
        self._clear_noise_cache()

    def _clear_noise_cache(self):
        if self._activation_noise_cache is not None:
            self._activation_noise_cache.clear()
            self._activation_noise_cache_time = self._time

    @property
    @contextmanager
    def fixed_noise(self):
        """A context manager used to force multiple activations of a given chunk at the
        same time to use the same activation noise.

        .. warning::
            Use of ``fixed_noise`` is rarely appropriate, and easily leads to biologically
            implausible results. It is provided only for esoteric purposes. When its use
            is required it should be wrapped around the smallest fragment of code
            practical.

        >>> m = Memory()
        >>> m.learn(color="red")
        True
        >>> m.activation_history = []
        >>> m.retrieve()
        <Chunk 0000 {'color': 'red'}>
        >>> m.retrieve()
        <Chunk 0000 {'color': 'red'}>
        >>> pprint(m.activation_history, sort_dicts=False)
        [{'name': '0000',
          'creation_time': 0,
          'attributes': (('color', 'red'),),
          'references': (0,),
          'base_activation': 0.0,
          'activation_noise': 0.07779212346913301,
          'activation': 0.07779212346913301},
         {'name': '0000',
          'creation_time': 0,
          'attributes': (('color', 'red'),),
          'references': (0,),
          'base_activation': 0.0,
          'activation_noise': -0.015345110792246082,
          'activation': -0.015345110792246082}]
        >>> m.activation_history = []
        >>> with m.fixed_noise:
        ...     m.retrieve()
        ...     m.retrieve()
        ...
        <Chunk 0000 {'color': 'red'}>
        <Chunk 0000 {'color': 'red'}>
        >>> pprint(m.activation_history, sort_dicts=False)
        [{'name': '0000',
          'creation_time': 0,
          'attributes': (('color', 'red'),),
          'references': (0,),
          'base_activation': 0.0,
          'activation_noise': 0.8614281690342627,
          'activation': 0.8614281690342627},
         {'name': '0000',
          'creation_time': 0,
          'attributes': (('color', 'red'),),
          'references': (0,),
          'base_activation': 0.0,
          'activation_noise': 0.8614281690342627,
          'activation': 0.8614281690342627}]
        """
        self._activation_noise_cache = {}
        self._activation_noise_cache_time = self._time
        try:
            yield self
        finally:
            self._activation_noise_cache = None
            self._activation_noise_cache_time = None

    @property
    def time(self):
        """This Memory's current time.
        Time in PyACTUp is a dimensionless quantity, the interpretation of which is at the
        discretion of the modeler.
        """
        return self._time

    def advance(self, amount=1):
        """Adds the given *amount* to this Memory's time, and returns the new, current time.
        Raises a :exc:`ValueError` if *amount* is negative, or not a real number.
        """
        if amount < 0:
            raise ValueError(f"Time cannot be advanced backward ({amount})")
        if amount:
            self._time += amount
            self._clear_noise_cache()
        return self._time

    @property
    @contextmanager
    def current_time(self):
        """A context manager used to allow reverting to the current time after advancing
        it and simiulating retrievals or similar operations in the future.

        .. warning::
            It is rarely appropriate to use ``current_time``. When it is used, care should
            be taken to avoid creating biologically implausible models. Also, learning
            within a ``current_time`` context will typically lead to tears as having
            chunks created or reinforced in the future results in failures of attempts to
            retrieve them.

        >>> m = Memory(temperature=1, noise=0)
        >>> m.learn(size=1)
        True
        >>> m.advance(10)
        11
        >>> m.learn(size=10)
        True
        >>> m.blend("size")
        7.983916860341838
        >>> with m.current_time as t:
        ...     m.advance(10_000)
        ...     m.blend("size")
        ...     (t, m.time)
        ...
        10012
        5.501236696240907
        (12, 10012)
        >>> m.time
        12
        """
        old = self._time
        try:
            yield old
        finally:
            self._time = old

    @property
    def learning_time_increment(self):
        """The default amount of time to :meth:`advance` by after performing a learn
        operation. By default this is ``1``. Attempting  to set this to a negative value
        raises a :exc:`ValueError`."""
        return self._learning_time_increment

    @learning_time_increment.setter
    def learning_time_increment(self, value):
        if value is None:
            value = 0
        if value < 0:
            raise ValueError(f"The learning_time_increment cannot be negative ({value})")
        self._learning_time_increment = value

    @property
    def retrieval_time_increment(self):
        """The default amount of time to :meth:`advance` by before performing a retrieval
        or blending operation. By default this is zero. Attempting to set this to a
        negative values raise a :exc:`ValueError`."""
        return self._retrieval_time_increment

    @retrieval_time_increment.setter
    def retrieval_time_increment(self, value):
        if value is None:
            value = 0
        if value < 0:
            raise ValueError(f"The retrieval_time_increment cannot be negative ({value})")
        self._retrieval_time_increment = value

    @property
    def noise(self):
        """The amount of noise to add during chunk activation computation.
        This is typically a positive, floating point, number between about 0.1 and 1.5.
        It defaults to 0.25.
        If zero, no noise is added during activation computation.
        If an explicit :attr:`temperature` is not set, the value of noise is also used
        to compute a default temperature for blending computations.
        Attempting to set :attr:`noise` to a negative number raises a :exc:`ValueError`.
        """
        return self._noise

    @noise.setter
    def noise(self, value):
        if value < 0:
            raise ValueError(f"The noise, {value}, must not be negative")
        if self._temperature_param is None:
            t = Memory._validate_temperature(None, value)
            if not t:
                warn(f"Setting noise to {value} will make the temperature too low; setting temperature to 1")
                self.temperature = 1
            else:
                self._temperature = t
        if value != self._noise:
            self._noise = value
            self._clear_noise_cache()

    @property
    def decay(self):
        """Controls the rate at which activation for previously chunks in memory decay with the passage of time.
        Time in this sense is dimensionless.
        The :attr:`decay` is typically between about 0.1 and 2.0.
        The default value is 0.5. If zero memory does not decay.
        If set to ``None`` no base level activation is computed or used; note that this is
        significantly different than setting it to zero which causes base level activation
        to still be computed and used, but with no decay.
        Attempting to set it to a negative number raises a :exc:`ValueError`.
        It must be less one 1 if this memory's :attr:`optimized_learning` parameter is set.
        """
        return self._decay

    @decay.setter
    def decay(self, value):
        if value is not None:
            if value < 0:
                raise ValueError(f"The decay, {value}, must not be negative")
            if value < 1:
                self._ln_1_mius_d = math.log(1 - value)
            elif self._optimized_learning:
                self._ln_1_mius_d = "illegal value" # ensure error it attempt to use this
                raise ValueError(f"The decay, {value}, must be less than one if optimized_learning is True")
        self._ln_cache = [None]*LN_CACHE_SIZE
        self._decay = value
        for c in self.values():
            c._clear_base_activation()

    @property
    def temperature(self):
        """The temperature parameter used for blending values.
        If ``None``, the default, the square root of 2 times the value of
        :attr:`noise` will be used. If the temperature is too close to zero, which
        can also happen if it is ``None`` and the :attr:`noise` is too low, or negative, a
        :exc:`ValueError` is raised.
        """
        return self._temperature_param

    _SQRT_2 = math.sqrt(2)

    @temperature.setter
    def temperature(self, value):
        if value is None or value is False:
            value = None
        else:
            value = float(value)
        t = Memory._validate_temperature(value, self._noise)
        if not t:
            if value is None:
                raise ValueError(f"The noise, {self._noise}, is too low to for the temperature to be set to None.")
            else:
                raise ValueError(f"The temperature, {value}, must not be less than {MINIMUM_TEMPERATURE}.")
        self._temperature_param = value
        self._temperature = t

    @staticmethod
    def _validate_temperature(temperature, noise):
        if temperature is not None:
            t = temperature
        else:
            t = Memory._SQRT_2 * noise
        if t < MINIMUM_TEMPERATURE:
            return None
        else:
            return t

    @property
    def threshold(self):
        """The minimum activation value required for a retrieval.
        If ``None`` there is no minimum activation required.
        The default value is ``-10``.
        Attempting to set the ``threshold`` to a value that is neither ``None`` nor a
        real number raises a :exc:`ValueError`.

        While for the likelihoods of retrieval the values of attr:`time` are normally
        scale free, not depending upon the magnitudes of attr:`time`, but rather the
        ratios of various times, the attr:`threshold` is sensitive to the actual
        magnitude. Suitable care should be exercised when adjusting it.
        """
        if self._threshold == -sys.float_info.max:
            return None
        else:
            return self._threshold

    @threshold.setter
    def threshold(self, value):
        if value is None or value is False:
            self._threshold = -sys.float_info.max
        else:
            self._threshold = float(value)

    @property
    def mismatch(self):
        """The mismatch penalty applied to partially matching values when computing activations.
        If ``None`` no partial matching is done.
        Otherwise any defined similarity functions (see :func:`set_similarity_function`)
        are called as necessary, and
        the resulting values are multiplied by the mismatch penalty and subtracted
        from the activation.

        Attributes for which no similarity function has been defined are always compared
        exactly, and chunks not matching on this attributes are not included at all in the
        corresponding partial retrievals or blending operations.

        Attempting to set this parameter to a value other than ``None`` or a real number
        raises a :exc:`ValueError`.
        """
        return self._mismatch

    @mismatch.setter
    def mismatch(self, value):
        if value is None or value is False:
            self._mismatch = None
        elif value < 0:
            raise ValueError(f"The mismatch penalty, {value}, must not be negative")
        else:
            self._mismatch = float(value)

    @property
    def activation_history(self):
        """A :class:`MutableSequence`, typically a :class:`list`,  into which details of the computations underlying PyACTUp operation are appended.
        If ``None``, the default, no such details are collected.
        In addition to activation computations, the resulting retrieval probabilities are
        also collected for blending operations.
        The details collected are presented as dictionaries.
        The ``references`` entries in these dictionaries are sequences of times the
        corresponding chunks were learned, if :attr:`optimizied_learning` is off, and
        otherwise are counts of the number of times they have been learned.

        If PyACTUp is being using in a loop, the details collected will likely become
        voluminous. It is usually best to clear them frequently, such as on each
        iteration.

        Attempting to set :attr:`activation_history` to anything but ``None`` or a
        :class:`MutableSequence` raises a :exc:`ValueError`.

        >>> m = Memory()
        >>> m.learn(color="red", size=3)
        True
        >>> m.learn(color="red", size=5)
        True
        >>> m.activation_history = []
        >>> m.blend("size", color="red")
        4.027391084562462
        >>> pprint(m.activation_history, sort_dicts=False)
        [{'name': '0000',
          'creation_time': 0,
          'attributes': (('color', 'red'), ('size', 3)),
          'references': (0,),
          'base_activation': -0.3465735902799726,
          'activation_noise': 0.4750912862904178,
          'activation': 0.12851769601044521,
          'retrieval_probability': 0.48630445771876907},
         {'name': '0001',
          'creation_time': 1,
          'attributes': (('color', 'red'), ('size', 5)),
          'references': (1,),
          'base_activation': 0.0,
          'activation_noise': 0.14789096368864968,
          'activation': 0.14789096368864968,
          'retrieval_probability': 0.5136955422812309}]
        """
        return self._activation_history

    @activation_history.setter
    def activation_history(self, value):
        if value is None or value is False:
            self._activation_history = None
        elif isinstance(value, abc.MutableSequence):
            self._activation_history = value
        else:
            raise ValueError(
                f"A value assigned to activation_history must be a MutableSequence ({value}).")

    @property
    def chunks(self):
        """ Returns a :class:`list` of the :class:`Chunk`s contained in this :class:`Memory`.
        """
        return list(self.values())

    def print_chunks(self, file=sys.stdout, pretty=True):
        """Prints descriptions of all the :class:`Chunk` objects contained in this :class:`Memory`.
        The descriptions are printed to *file*, which defaults to the standard output. If
        *file* is not an open text file it should be a string naming a file to be opened
        for writing.

        If *pretty* is true, the default, a format intended for reading by humans is used.
        Otherwise comma separated values (CSV) format, more suitable for importing into
        spreadsheets, numpy, and the like, is used.

        .. warning::
            This method is intended as a debugging aid, and generally is not suitable for
            use as a part of models.
        """
        if not self:
            return
        if isinstance(file, io.TextIOBase):
            data = [{"chunk name": c._name,
                     "chunk contents": dict(k).__repr__()[1:-1],
                     "chunk created at": c._creation,
                     "chunk references": (c.references if isinstance(c.references, Number)
                                          else c.references.__repr__()[1:-1])}
                    for k, c in self.items()]
            if pretty:
                tab = PrettyTable()
                tab.field_names = data[0].keys()
                for d in data:
                    tab.add_row(d.values())
                print(tab, file=file, flush=True)
            else:
                w = csv.DictWriter(file, data[0].keys())
                w.writeheader()
                for d in data:
                    w.writerow(d)
        else:
            with open(file, "w+", newline=(None if pretty else "")) as f:
                self.print_chunks(f, pretty)

    @property
    def optimized_learning(self):

        """A boolean indicating whether or not this Memory is configured to use optimized learning.
        Cannot be set directly, but can be changed when calling :meth:`reset`.

        .. warning::
            Care should be taken when using optimized learning as operations such as
            ``retrieve`` that depend upon activation will not longer raise an exception if
            they are called when ``advance`` has not been called after ``learn``, possibly
            producing biologically implausible results.
        """
        return self._optimized_learning

    _use_actr_similarity = False
    _minimum_similarity = 0
    _maximum_similarity = 1
    _similarity_functions = {}
    _similarity_cache = pylru.lrucache(SIMILARITY_CACHE_SIZE)

    # possibly update docstrings for retrieve() and blend() to reflect new similarity stuff,
    # or maybe put it all in set_similarity_function() and/or mismatch?
    def _similarity(self, x, y, attribute):
        # always returns the "natural" similarity
        # returns None if similarity is inapplicatble to attribute
        if x == y:
            return 1
        fn = self._similarity_functions.get(attribute)
        if fn is True:
            return 0
        elif fn:
            signature = (x, y, attribute)
            result = self._similarity_cache.get(signature)
            if result is not None:
                return result
            result = self._similarity_cache.get((y, x, attribute))
            if result is not None:
                return result
            result = fn(x, y)
            if result < Memory._minimum_similarity:
                warn(f"similarity value is less than the minimum allowed, {Memory._minimum_similarity}, so that minimum value is being used instead")
                result = Memory._minimum_similarity
            elif result > Memory._maximum_similarity:
                warn(f"similarity value is greater than the maximum allowed, {Memory._maximum_similarity}, so that maximum value is being used instead")
                result = Memory._maximum_similarity
            if Memory._use_actr_similarity:
                result += 1
            self._similarity_cache[signature] = result
            return result

    def learn(self, advance=None, **kwargs):
        """Adds, or reinforces, a chunk in this Memory with the attributes specified by *kwargs*.
        The attributes, or slots, of a chunk are described using Python keyword arguments.
        The attribute names must conform to the usual Python variable name syntax, and may
        be neither Python keywords nor the names of optional arguments to :meth:`learn`,
        :meth:`retrieve` or :meth:`blend`: *partial*, *reheard* or *advance*. Their
        values must be :class:`Hashable`.

        Returns ``True`` if a new chunk has been created, and ``False`` if instead an
        already existing chunk has been re-experienced and thus reinforced.

        After learning the relevant chunk, :meth:`advance` is called with an argument of
        *advance*. If *advance* has not been supplied it defaults to the current value of
        :attr:`learning_time_increment`; unless this has been changed by the programmer
        this default value is ``1``. If this *advance* is zero then time must be advanced
        by the programmer with :meth:`advance` following any calls to ``learn`` before
        calling :meth:`retrieve` or :meth:`blend`. Otherwise the chunk learned at this
        time would have infinite activation.

        Raises a :exc:`TypeError` if an attempt is made to learn an attribute value that
        is not :class:`Hashable`.

        >>> m = Memory()
        >>> m.learn(color="red", size=4)
        True
        >>> m.learn(color="blue", size=4)
        True
        >>> m.learn(color="red", size=4)
        False
        >>> m.retrieve(color="red")
        <Chunk 0000 {'color': 'red', 'size': 4}>

        """
        if not kwargs:
            raise ValueError(f"No attributes to learn")
        created = False
        signature = tuple(sorted(kwargs.items()))
        chunk = self.get(signature)
        if not chunk:
            chunk = Chunk(self, kwargs)
            self[signature] = chunk
            created = True
        self._cite(chunk)
        self._advance(advance, self._learning_time_increment)
        return created

    def _advance(self, argument, default):
        old = self._time
        self.advance(argument if argument is not None else default)
        return old

    def _cite(self, chunk):
        if not self._optimized_learning:
            if chunk._reference_count >= chunk._references.size:
                chunk._references.resize(2 * chunk._references.size, refcheck=False)
            chunk._references[chunk._reference_count] = self._time
        chunk._reference_count += 1
        chunk._base_activation_time = None

    def forget(self, when, **kwargs):
        """Undoes the operation of a previous call to :meth:`learn`.

        .. warning::
            Normally this method should not be used. It does not correspond to a
            biologically plausible process, and is only provided for esoteric purposes.

        The *kwargs* should be those supplied fro the :meth:`learn` operation to be
        undone, and *when* should be the time that was current when the operation was
        performed. Returns ``True`` if it successfully undoes such an operation, and
        ``False`` otherwise.
        """
        if not kwargs:
            raise ValueError(f"No attributes to forget")
        signature = tuple(sorted(kwargs.items()))
        chunk = self.get(signature)
        if not chunk:
            return False
        if not self._optimized_learning:
            try:
                i = np.where(chunk._references == when)[0][0]
            except IndexError:
                return False
            if i < chunk._reference_count:
                chunk._references[i:chunk._reference_count-1] = chunk._references[i+1:chunk._reference_count]
        elif when < chunk._creation:
            return False
        elif when == chunk._creation and chunk._reference_count > 1:
            raise RuntimeError("Can't meaningfully forget a chunk at its creation time with optimized learning")
        chunk._reference_count -= 1
        if not chunk._reference_count:
            del self[signature]
        return True

    def retrieve(self, partial=False, rehearse=False, advance=None, **kwargs):
        """Returns the chunk matching the *kwargs* that has the highest activation greater than this Memory's :attr:`threshold`.
        If there is no such matching chunk returns ``None``.
        Normally only retrieves chunks exactly matching the *kwargs*; if *partial* is
        ``True`` it also retrieves those only approximately matching, using similarity
        (see :func:`set_similarity_function`) and :attr:`mismatch` to determine closeness
        of match.

        Before performing the retrieval :meth:`advance` is called with the value of
        *advance* as its argument. If *advance* is not supplied the current value
        of :attr:`retrieval_time_increment` is used; unless changed by the programmer this
        default value is zero. The advance of time does not occur if an error is raised
        when attempting to perform the retrieval.

        If *rehearse* is supplied and true it also reinforces this chunk at the current
        time. No chunk is reinforced if retrieve returns ``None``.

        The returned chunk is a dictionary-like object, and its attributes can be
        extracted with Python's usual subscript notation.

        >>> m = Memory()
        >>> m.learn(widget="thromdibulator", color="red", size=2)
        True
        >>> m.learn(widget="snackleizer", color="blue", size=1)
        True
        >>> m.retrieve(color="blue")["widget"]
        'snackleizer'
        """
        old = self._advance(advance, self._retrieval_time_increment)
        try:
            result = self._partial_match(kwargs) if partial else self._exact_match(kwargs)
            if rehearse and result:
                self._cite(result)
            old = None
            return result
        finally:
            if old is not None:
                # Don't advance if there's an error or for some other reason we don't
                # finish normally; note that the noise cache is still cleared, though.
                self._time = old

    def _exact_match(self, conditions):
        # Returns a single chunk matching the given slots and values, that has the
        # highest activation greater than the threshold parameter. If there are no
        # such chunks returns None.
        best_chunk = None
        best_activation = self._threshold
        for chunk in self.values():
            if not conditions.keys() <= chunk.keys():
                continue
            for key, value in conditions.items():
                if chunk[key] != value:
                    break
            else:   # this matches the for, NOT the if
                a = chunk._activation()
                if a >= best_activation:
                    best_chunk = chunk
                    best_activation = a
        return best_chunk

    def _make_noise(self, chunk):
        if not self._noise:
            return 0
        if self._activation_noise_cache is not None:
            result = self._activation_noise_cache.get(chunk._name)
        else:
            result = None
        if result is None:
            if self._next_noise_value >= NOISE_VALUES_SIZE:
                self._noise_values = self._rng.logistic(scale=self._noise, size=NOISE_VALUES_SIZE)
                self._next_noise_value = 0
            result = self._noise_values[self._next_noise_value]
            self._next_noise_value += 1
        if self._activation_noise_cache is not None:
            self._activation_noise_cache[chunk._name] = result
        return result

    class _Activations(abc.Iterable):

        def __init__(self, memory, conditions):
            self._memory = memory
            self._conditions = conditions

        def __iter__(self):
            self._chunks = self._memory.values().__iter__()
            return self

        def __next__(self):
            while True:
                chunk = self._chunks.__next__()             # pass on up the Stop Iteration
                if self._conditions.keys() <= chunk.keys(): # subset
                    if self._memory._mismatch is None:
                        exact = self._conditions.keys()
                        partial = []
                    else:
                        exact = []
                        partial = []
                        for c in self._conditions.keys():
                            if Memory._similarity_functions.get(c):
                                partial.append(c)
                            else:
                                exact.append(c)
                    if not all(chunk[a] == self._conditions[a] for a in exact):
                        continue
                    activation = chunk._activation(True)
                    if self._memory._mismatch is None:
                        if self._memory._activation_history is not None:
                            self._memory._activation_history[-1]["activation"] = activation
                        return (chunk, activation)
                    mismatch = (self._memory._mismatch
                                * sum(self._memory._similarity(self._conditions[a], chunk[a], a) - 1
                                      for a in partial))
                    total = activation + mismatch
                    if self._memory._activation_history is not None:
                        history = self._memory._activation_history[-1]
                        history["mismatch"] = mismatch
                        history["activation"] = total
                    return (chunk, total)


    def _activations(self, conditions):
         return self._Activations(self, conditions)

    def _partial_match(self, conditions):
        best_chunks = []
        best_activation = self._threshold
        for chunk, activation in self._activations(conditions):
            if activation > best_activation:
                best_chunks = [chunk]
                best_activation = activation
            elif activation == best_activation:
                best_chunks.add(chunk)
        return random.choice(best_chunks) if best_chunks else None

    def blend(self, outcome_attribute, advance=None, **kwargs):
        """Returns a blended value for the given attribute of those chunks matching *kwargs*, and which contains *outcome_attribute*.
        Returns ``None`` if there are no matching chunks that contains
        *outcome_attribute*. If any matching chunk has a value of *outcome_attribute*
        value that is not a real number a :exc:`TypeError` is raised.

        Before performing the blending operation :meth:`advance` is called with the value
        of *advance* as its argument. If *advance* is not supplied the current value
        of :attr:`retrieval_time_increment` is used; unless changed by the programmer this
        default value is zero. The advance of time does not occur if an error is raised
        when attempting to perform the blending operation.

        >>> m = Memory()
        >>> m.learn(color="red", size=2)
        True
        >>> m.learn(color="blue", size=30)
        True
        >>> m.learn(color="red", size=1)
        True
        >>> m.blend("size", color="red")
        1.1548387620911693
        """
        old = self._advance(advance, self._retrieval_time_increment)
        try:
            weights = 0.0
            weighted_outcomes = 0.0
            if self._activation_history is not None:
                chunk_weights = []
            for chunk, activation in self._activations(kwargs):
                if outcome_attribute not in chunk:
                    continue
                weight = math.exp(activation / self._temperature)
                if self._activation_history is not None:
                    chunk_weights.append((self._activation_history[-1], weight))
                weights += weight
                weighted_outcomes += weight * chunk[outcome_attribute]
            if self._activation_history is not None:
                for history, w in chunk_weights:
                    try:
                        history["retrieval_probability"] = w / weights
                    except ZeroDivisionError:
                        history["retrieval_probability"] = None
            try:
                result = weighted_outcomes / weights
                old = None
                return result
            except ZeroDivisionError:
                return None
        finally:
            if old is not None:
                # Don't advance if there's an error or for some other reason we don't
                # finish normally; note that the noise cache is still cleared, though.
                self._time = old

    def best_blend(self, outcome_attribute, iterable, select_attribute=None, advance=None, minimize=False):
        """Returns two values (as a 2-tuple), describing the extreme blended value of the *outcome_attribute* over the values provided by *iterable*.
        The extreme value is normally the maximum, but can be made the minimum by setting
        *minimize* to True. The values returned by *iterable* should be dictionary-like
        object that can be passed as the *kwargs* argument to :meth:`blend`. The first
        return value is the *kwargs* value producing the best blended value, and second is
        that blended value. If there is a tie, with two or more *kwargs* values all
        producing the same, best blended value, then one of them is chosen randomly. If
        none of the values from *iterable* result in blended values of *outcome_attribute*
        then both return values are ``None``.

        This operation is particularly useful for building Instance Based Learning models.

        For the common case where *iterable* iterates over only the values of a single
        slot the *select_attribute* parameter may be used to simplify the iteration. If
        *select_attribute* is supplied and is not ``None`` then *iterable* should produce
        values of that slot instead of dictionary-like objects. Similarly the first return
        value will be the slot value rather than a dictionary-like object. The end of the
        example below demonstrates this.

        Before performing the blending operations :meth:`advance` is called, only once,
        with the value of *advance* as its argument. If *advance* is not supplied the
        current value of :attr:`retrieval_time_increment` is used; unless changed by the
        programmer this default value is zero. The advance of time does not occur if an
        error is raised when attempting to perform a blending operation.

        >>> m = Memory()
        >>> m.learn(color="red", utility=1)
        True
        >>> m.learn(color="blue", utility=2)
        True
        >>> m.learn(color="red", utility=1.8)
        True
        >>> m.learn(color="blue", utility=0.9)
        True
        >>> m.best_blend("utility", ({"color": c} for c in ("red", "blue")))
        ({'color': 'blue'}, 1.4868)
        >>> m.learn(color="blue", utility=-1)
        True
        >>> m.best_blend("utility", ("red", "blue"), "color")
        ('red', 1.2127)

        """
        comparator = operator.gt if not minimize else operator.lt
        old = self._advance(advance, self._retrieval_time_increment)
        try:
            best_value = -math.inf if not minimize else math.inf
            best_args = []
            for thing in iterable:
                if select_attribute is not None:
                    kwargs = { select_attribute : thing }
                else:
                    kwargs = thing
                value = self.blend(outcome_attribute, advance=0, **kwargs)
                if value is None:
                    pass
                elif value == best_value:
                    best_args.append(kwargs)
                elif comparator(value, best_value):
                    best_args = [ kwargs ]
                    best_value = value
            old = None
            if best_args:
                result = random.choice(best_args)
                if select_attribute is not None:
                    result = result[select_attribute]
                return result, best_value
            else:
                return None, None
        finally:
            if old is not None:
                # Don't advance if there's an error or for some other reason we don't
                # finish normally; note that the noise cache is still cleared, though.
                self._time = old


def use_actr_similarity(value=None):
    """Whether to use "natural" similarity values, or traditional ACT-R ones.
    PyACTUp normally uses a "natural" representation of similarities, where two values
    being completely similar, identical, has a value of one; and being completely
    dissimilar has a value of zero; with various other degrees of similarity being
    positive, real numbers less than one. Traditionally ACT-R instead uses a range of
    similarities with the most dissimilar being a negative number, usually -1, and
    completely similar being zero.

    If the argument is ``False`` or ``True`` is sets the ACT-R traditional behavior
    on or off, and returns it. With no arguments it returns the current value.
    """
    if value is not None:
        Memory._similarity_cache.clear()
        if value:
            Memory._minimum_similarity = -1
            Memory._maximum_similarity =  0
        else:
            Memory._minimum_similarity =  0
            Memory._maximum_similarity =  1
        Memory._use_actr_similarity = bool(value)
    return Memory._use_actr_similarity

def set_similarity_function(function, *slots):
    """Assigns a similarity function to be used when comparing attribute values with the given names.
    The function should take two arguments, and return a real number between 0 and 1,
    inclusive.
    The function should be commutative; that is, if called with the same arguments
    in the reverse order, it should return the same value.
    It should also be stateless, always returning the same values if passed
    the same arguments.
    No error is raised if either of these constraints is violated, but the results
    will, in most cases, be meaningless if they are.

    If ``True`` is supplied as the *function* a default similarity function is used that
    returns one if its two arguments are ``==`` and zero otherwise.

    >>> def f(x, y):
    ...     if y < x:
    ...         return f(y, x)
    ...     return 1 - (y - x) / y
    >>> set_similarity_function(f, "length", "width")
    """
    for s in slots:
        if callable(function):
            Memory._similarity_functions[s] = function
        elif function:
            Memory._similarity_functions[s] = True
        elif s in Memory._similarity_functions:
            del Memory._similarity_functions[s]
    Memory._similarity_cache.clear()


class Chunk(dict):

    __slots__ = ["_name", "_memory", "_creation", "_references", "_reference_count",
                 "_base_activation_time", "_base_activation"]

    _name_counter = 0;

    def __init__(self, memory, content):
        self._name = f"{Chunk._name_counter:04d}"
        Chunk._name_counter += 1
        self._memory = memory
        self.update(content)
        self._creation = memory._time
        self._references = np.empty(1, dtype=int)
        self._reference_count = 0
        self._base_activation_time = None
        self._base_activation = None

    def __repr__(self):
        return "<Chunk {} {} {}>".format(self._name, dict(self), self._reference_count)

    def __str__(self):
        return self._name

    @property
    def references(self):
        """Returns when this :class:`Chunk` has been reinforced.
        The type of the value returned depends upon whether or not the :class:`Memory`
        containing it is using optimized learning. If it is this is an integer, the number
        of times this :class:`Chunk` has been reinforced, and otherwise is a list of times
        at which it was reinforced.
        """
        if self._memory._optimized_learning:
            return self._reference_count
        else:
            return list(self._references[:self._reference_count])

    def _activation(self, for_partial=False):
        # Does not include the mismatch penalty component, that's handled by the caller.
        base = self._get_base_activation() if self._memory._decay is not None else 0
        noise = self._memory._make_noise(self)
        result = base + noise
        if self._memory._activation_history is not None:
            history = {"name": self._name,
                       "creation_time": self._creation,
                       "attributes": tuple(self.items()),
                       "references": (self._reference_count
                                      if self._memory.optimized_learning
                                      else tuple(self._references[:self._reference_count])),
                       "base_activation": base,
                       "activation_noise": noise}
            if not for_partial:
                history["activation"] = result
            self._memory._activation_history.append(history)
        return result

    # Note that memoizing ln doesn't make much difference, but it does speed
    # things up a tiny bit, most noticeably under PyPy.
    def _cached_ln(self, arg):
        try:
            result = self._memory._ln_cache[arg]
        except (IndexError, TypeError):
            return math.log(arg)
        if result is None:
            result = math.log(arg)
            self._memory._ln_cache[arg] = result
        return result

    def _get_base_activation(self):
        if self._base_activation_time != self._memory.time:
            err = None
            if self._memory._optimized_learning:
                try:
                    self._base_activation = (self._cached_ln(self._reference_count)
                                             - self._memory._ln_1_mius_d
                                             - self._memory._decay * self._cached_ln(self._memory._time - self._creation))
                except ValueError as e:
                    err = e
            else:
                base = np.sum((self._memory._time - self._references[0:self._reference_count])
                              ** -self._memory._decay)
                if np.isfinite(base):
                    self._base_activation = math.log(base)
                else:
                    err = RuntimeError(f"Non-finite value {base} encounterd when computing base activation")
            if err:
                if self._memory._time <= self._creation:
                    raise RuntimeError("Can't compute activation of a chunk at or before the time it was created")
                elif (not self._memory._optimized_learning
                      and self._references[-1] >= self._memory._time):
                    raise RuntimeError("Can't compute activation of a chunk at or before the time of its most recent reference")
                else:
                    raise err
            self._base_activation_time = self._memory.time
        return self._base_activation

    def _clear_base_activation(self):
        self._base_activation_time = None



# Local variables:
# fill-column: 90
# End:
