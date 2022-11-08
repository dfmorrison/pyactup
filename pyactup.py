# Copyright (c) 2018-2022 Carnegie Mellon University
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
applications. Its creation was inspired by the ACT-UP cognitive modeling toolbox.

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

__version__ = "2.0.2"

if "dev" in __version__:
    print("PyACTUp version", __version__)

import collections.abc as abc
import csv
import io
import math
import numpy as np
import numpy.ma as ma
import operator
import random
import sys

from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager
from itertools import count
from numbers import Real
from prettytable import PrettyTable
from pylru import lrucache
from warnings import warn

__all__ = ["Memory"]

DEFAULT_NOISE = 0.25
DEFAULT_DECAY = 0.5

MINIMUM_TEMPERATURE = 0.01

REFERENCES_FACTOR = 4
SIMILARITY_CACHE_SIZE = 10_000
MAXIMUM_RANDOM_SEED = 2**62

class Memory(dict):
    """A cognitive entity containing a collection of learned things, its chunks.
    A ``Memory`` object also contains a current time, which can be queried as the
    :attr:`time` property.

    The number of distinct chunks a ``Memory`` contains can be determined with Python's
    usual :func:`len` function.

    A ``Memory`` has several parameters controlling its behavior: :attr:`noise`,
    :attr:`decay`, :attr:`temperature`, :attr:`threshold`, :attr:`mismatch`, and
    :attr:`optimized_learning`, and :attr:`use_actr_similarity`. All can be queried and
    set as properties on the ``Memory`` object. When creating a ``Memory`` object their
    initial values can be supplied as parameters.

    A ``Memory`` object can be serialized with `pickle
    <https://docs.python.org/3.8/library/pickle.html>`_ allowing Memory objects to be
    saved to and restored from persistent storage, so long as any similarity functions it
    contains are defined at the top level of a module using ``def``. Note that attempts to
    pickle a Memory object containing a similarity function defined as a lambda function,
    or as an inner function, will cause a :exc:`Exception` to be raised. And note further
    that pickle only includes the function name in the pickled object, not its definition.
    Also, if the contents of a ``Memory`` object are sufficiently complicated it may be
    necessary to raise Python's recursion limit with
    `sys.setrecusionlimit <https://docs.python.org/3.8/library/sys.html#sys.setrecursionlimit>`_.

    A common use case for PyACTUp involves all of the chunks in a ``Memory`` having the
    same attributes, and some of those attributes are always used, by matching exactly,
    not partially, some of those attributes. The ``index``keyword argument declares that
    such a set of attributes is present, and can result in significant performance
    improvements for models with a *very* large number of chunks. The value of this
    keyword argument should be a list of attribute names. As a convenience, if none of the
    attribute names contains commas or spaces, a string maybe used instead of a list, the
    attribute names being separated by spaces or commas; either spaces or commas must be
    used, not a mixture. For example, both ``index="decision utility"`` and
    ``index="decision,utiliy"`` are equivalent to ``index=["decision", "utility"]``. A
    list of he attributes in a :class:`Memory`'s *index* can be retrieved with the
    :attr:`index` property. If the ``Memory`` is empty, containing no chunks, the *index*
    can be modified by setting that property, but otherwise the *index* cannot be changed
    after the ``Memory`` was created. `All chunks in a ``Memory`` with an *index* must
    contain values for all the attributes listed in the *index*; if any are omitted in the
    argument to :meth:`learn` they will be automatically added with a value of ``None``.

    If, when creating a ``Memory`` object, any of the various parameters have unsupported
    values an :exc:`Exception` will be raised. See the documentation for the various
    properties that can be used for setting these parameters for further details about
    what values are or are not supported.

    """

    def __init__(self,
                 noise=DEFAULT_NOISE,
                 decay=DEFAULT_DECAY,
                 temperature=None,
                 threshold=None,
                 mismatch=None,
                 optimized_learning=False,
                 use_actr_similarity=False,
                 index=None):
        self._fixed_noise = None
        self._fixed_noise_time = None
        self._temperature_param = 1 # will be reset below, but is needed for noise assignment
        self._noise = None
        self._decay = None
        self._optimized_learning = None
        self._use_actr_similarity = False
        self._minimum_similarity = 0
        self._maximum_similarity = 1
        self._similarities = defaultdict(Similarity)
        self.noise = noise
        self.decay = decay
        if temperature is None and not self._validate_temperature(None, noise):
            warn(f"A noise of {noise} and temperature of None will make the temperature "
                 f"too low; setting temperature to 1")
            self.temperature = 1
        else:
            self.temperature = temperature
        self.threshold = threshold
        self.mismatch = mismatch
        self.optimized_learning = optimized_learning
        self.use_actr_similarity = use_actr_similarity
        self._slot_name_index = defaultdict(list)
        self._indexed_attributes = set()
        self._index = defaultdict(list)
        self.index = index
        self._activation_history = None
        # Initialize the noise RNG from the parent Python RNG, in case the latter gets seeded for determinancy.
        self._rng = np.random.default_rng([random.randint(0, MAXIMUM_RANDOM_SEED) for i in range(16)])
        self.reset()

    def __repr__(self):
        return f"<Memory {id(self)}: {list(self._indexed_attributes)}, {len(self)}, {self._time}>"

    def reset(self, preserve_prepopulated=False, index=None):
        """Deletes this :class:`Memory`'s chunks and resets its time to zero.
        If *preserve_prepopulated* is ``False`` it deletes all chunks; if it is ``True``
        it deletes all chunk references later than time zero, completely deleting those
        chunks that were created at a time other than zero. If *index* is supplied it
        sets the :class:`Memory`'s index to that value.
        """
        if preserve_prepopulated:
            preserved = {k: v for k, v in self.items() if v._creation == 0}
        self.clear()
        self._slot_name_index.clear()
        self._index.clear()
        self._clear_fixed_noise()
        self._activation_history = None
        self._time = 0
        if index is not None:
            self.index = index
        if preserve_prepopulated:
            for k, v in preserved.items():
                v._references = np.empty(1, dtype=np.int32) if self._optimized_learning else np.array([0])
                v._reference_count = 1
                self[k] = v
                self._slot_name_index[frozenset(v.keys())].append(v)

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
        >>> m.advance()
        1
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
        old_fixed_noise = self._fixed_noise
        old_fixed_noise_time = self._fixed_noise_time
        try:
            if self._fixed_noise is None:
                self._fixed_noise = dict()
                self._fixed_noise_time = self._time
            yield self
        finally:
            self._fixed_noise = old_fixed_noise
            self._fixed_noise_time = old_fixed_noise_time

    def _clear_fixed_noise(self):
        if self._fixed_noise:
            self._fixed_noise.clear()
            self._fixed_noise_time = self._time

    @property
    def index(self):
        """A list of the attribute names in this ``Memory``'s index.
        If this :class:`Memory` is empty, containing no chunks, this can also be set,
        using the same syntax as in the :class:`Memory` constructor. However, if
        this ``Memory`` contains chunks an attempt to set the ``index`` will raise
        a :exc:`RuntimeError`.
        """
        return sorted(self._indexed_attributes)

    @index.setter
    def index(self, value):
        indexed_attributes = set(Memory._ensure_slot_names(value))
        if indexed_attributes == self._indexed_attributes:
            return
        if self:
            raise RuntimeError("Cannot set the index of a Memory after it contains chunks")
        assert not self._index and not self._slot_name_index
        self._indexed_attributes = indexed_attributes

    @property
    def time(self):
        """This ``Memory``'s current time.
        Time in PyACTUp is a dimensionless quantity, the interpretation of which is at the
        discretion of the modeler.
        """
        return self._time

    @time.setter
    def time(self, value):
        if not isinstance(value, Real):
            raise ValueError(f"Time {value} is not a real number")
        self._time = value
        if value != self._time:
            self._clear_fixed_noise()

    def advance(self, amount=1):
        """Adds the given *amount*, which defaults to 1, to this Memory's time, and returns the new, current time.
        Raises an :exc:`Exception` if *amount* is neither a real number nor ``None``.

        .. warning::
            While *amount* can be negative, this is rarely appropriate. Backward time can
            easily result in biologically implausible models, and attempts to perform
            retrievals or similar operations at times preceding those at which relevant
            chunks were created will result in infinite or complex valued base-level
            activations and raise an :exc:`Exception`.
        """
        if amount is not None:
            self.time += amount
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
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        TypeError: learn() got an unexpected keyword argument 'size'
        >>>
        >>>
        >>> m = Memory(temperature=1, noise=0)
        >>> m.learn({"size": 1})
        True
        >>> m.advance(10)
        10
        >>> m.learn({"size": 10})
        True
        >>> m.advance()
        11
        >>> m.blend("size")
        7.9150376886801395
        >>> with m.current_time as t:
        ...     m.advance(10_000)
        ...     m.blend("size")
        ...     (t, m.time)
        ...
        10011
        5.501124325474942
        (11, 10011)
        >>> m.time
        11
        """
        old = self._time
        try:
            yield old
        finally:
            self._time = old

    @property
    def noise(self):
        """The amount of noise to add during chunk activation computation.
        This is typically a positive, floating point, number between about 0.2 and 0.8.
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
            self._clear_fixed_noise()

    @property
    def decay(self):
        """Controls the rate at which activation for chunks in memory decay with the passage of time.
        Time in PyACTUp is dimensionless.
        The :attr:`decay` is typically between about 0.1 and 2.0.
        The default value is 0.5. If zero memory does not decay.
        If set to ``None`` no base-level activation is computed or used; note that this is
        significantly different than setting it to zero which causes base-level activation
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
            if value >= 1 and self._optimized_learning is not None:
                raise ValueError(f"The decay, {value}, must be less than one if optimized_learning is used")
        self._decay = value

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
        The default value is ``None``.
        Attempting to set the ``threshold`` to a value that is neither ``None`` nor a
        real number raises a :exc:`ValueError`.

        While for the likelihoods of retrieval the values of :attr:`time` are normally
        scale free, not depending upon the magnitudes of :attr:`time`, but rather the
        ratios of various times, the :attr:`threshold` is sensitive to the actual
        magnitude, and thus the units in which time is measured. Suitable care should be
        exercised when adjusting it.
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if value is None or value is False:
            self._threshold = None
        else:
            self._threshold = float(value)

    @property
    def mismatch(self):
        """The mismatch penalty applied to partially matching values when computing activations.
        If ``None`` no partial matching is done.
        Otherwise any defined similarity functions (see :meth:`similarity`)
        are called as necessary, and
        the resulting values are multiplied by the mismatch penalty and subtracted
        from the activation.

        Attributes for which no similarity function has been defined are always compared
        exactly, and chunks not matching on this attributes are not included at all in the
        corresponding partial retrievals or blending operations.

        While for the likelihoods of retrieval the values of :attr:`time` are normally
        scale free, not depending upon the magnitudes of :attr:`time`, but rather the
        ratios of various times, the :attr:`mismatch` is sensitive to the actual
        magnitude. Suitable care should be exercised when adjusting it.

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
    def optimized_learning(self):
        """Whether or not this Memory is configured to use the optimized learning approximation.
        If ``False``, the default, optimized learning is not used. If ``True`` is is used
        for all cases. If a positive integer, that number of the most recent rehearsals
        of a chunk are used exactly, with any older rehearsals having their contributions
        to the activation approximated.

        Attempting to set a value other than the above raises an :exc:`Exception`.

        Optimized learning can only be used if the :attr:`decay` is less than one.
        Attempting to set this parameter to ``True`` or an integer when :attr:`decay` is
        one or greater raises a :exc:`ValueError`.

        The value of this attribute can only be changed when the :class:`Memory` object
        does not contain any chunks, typically immediately after it is created or
        :meth:`reset`. Otherwise a :exc:`RuntimeError` is raised.

        .. warning::
            Care should be taken when using optimized learning as operations such as
            ``retrieve`` that depend upon activation may no longer raise an exception if
            they are called when ``advance`` has not been called after ``learn``, possibly
            producing biologically implausible results.
        """
        if self._optimized_learning is None:
            return False
        elif self._optimized_learning == 0:
            return True
        else:
            return self._optimized_learning

    @optimized_learning.setter
    def optimized_learning(self, value):
        if value is False or value is None:
            v = None
        elif value is True or value == 0:
            v = 0
        else:
            try:
                v = int(value)
            except:
                v = -1
            if v < 1:
                raise ValueError(f"The value of optimized learning must be a Boolean "
                                   f"or a positive integer, not {value}")
        if v is not None and self._decay and self._decay >= 1:
            raise ValueError(f"Optimized learning cannot be used when the decay, "
                               f"{self.decay}, is greater than or equal to one.")
        if self and v != self._optimized_learning:
            raise RuntimeError("Cannot change optimized learning for a Memory that "
                               "already contains chunks")
        self._optimized_learning = v

    @property
    def use_actr_similarity(self):
        """ Whether similarity computations for this :class:`Memory` use "natural" similarity values or traditional ACT-R ones.
        PyACTUp normally uses a "natural" representation of similarities, where two values
        being completely similar, identical, has a value of one; and being completely
        dissimilar has a value of zero; with various other degrees of similarity being
        positive, real numbers less than one. Traditionally ACT-R instead uses a range of
        similarities with the most dissimilar being a negative number, usually -1, and
        completely similar being zero. If the value of this :attr:`use_actr_similarity` is
        falsey, the default, natural similarities are used, and otherwise the tradional
        ACT-R ones.
        """
        return self._use_actr_similarity

    @use_actr_similarity.setter
    def use_actr_similarity(self, value):
        for s in self._similarities.values():
            s._cache.clear()
        if value:
            self._minimum_similarity = -1
            self._maximum_similarity =  0
        else:
            self._minimum_similarity =  0
            self._maximum_similarity =  1
        self._use_actr_similarity = bool(value)

    @property
    def activation_history(self):
        """A :class:`MutableSequence`, typically a :class:`list`, into which details of the computations underlying PyACTUp operation are appended.
        If ``None``, the default, no such details are collected.
        In addition to activation computations, the resulting retrieval probabilities are
        also collected for blending operations.
        The details collected are presented as dictionaries.
        As a convenience setting :attr:`activation_history` to ``True`` assigns a fresh,
        empty list as its value.

        If PyACTUp is being using in a loop, the details collected will likely become
        voluminous. It is usually best to clear them frequently, such as on each
        iteration.

        Attempting to set :attr:`activation_history` to anything but ``None``, ``True`` or
        a :class:`MutableSequence` raises a :exc:`ValueError`.

        >>> m = Memory()
        >>> m.learn({"color": "red", "size": 3})
        True
        >>> m.advance()
        1
        >>> m.learn({"color": "red", "size": 5})
        True
        >>> m.advance()
        2
        >>> m.activation_history = []
        >>> m.blend("size", {"color": "red"})
        4.810539051819914
        >>> pprint(m.activation_history, sort_dicts=False)
        [{'name': '0005',
          'creation_time': 0,
          'attributes': (('color', 'red'), ('size', 3)),
          'reference_count': 1,
          'references': [0],
          'base_level_activation': -0.3465735902799726,
          'activation_noise': -0.032318983984613185,
          'activation': -0.3788925742645858,
          'retrieval_probability': 0.09473047409004302},
         {'name': '0006',
          'creation_time': 1,
          'attributes': (('color', 'red'), ('size', 5)),
          'reference_count': 1,
          'references': [1],
          'base_level_activation': 0.0,
          'activation_noise': 0.4191470689622754,
          'activation': 0.4191470689622754,
          'retrieval_probability': 0.905269525909957}]

        """
        return self._activation_history

    @activation_history.setter
    def activation_history(self, value):
        if value is None or value is False:
            self._activation_history = None
        elif value is True:
            self._activation_history = list()
        elif isinstance(value, abc.MutableSequence):
            self._activation_history = value
        else:
            raise ValueError(
                f"A value assigned to activation_history must be a MutableSequence ({value}).")

    @property
    def chunks(self):
        """ Returns a :class:`list` of the :class:`Chunk` objects contained in this :class:`Memory`.
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

        If this :class:`Memory` is empty, not yet containing any chunks, nothing is
        printed, and no file is created.

        .. warning::
            The :meth:`print_chunks` method is intended as a debugging aid, and generally
            is not suitable for use as a part of a model.
        """
        if not self:
            return
        if isinstance(file, io.TextIOBase):
            data = [{"chunk name": c._name,
                     "chunk contents": dict(k).__repr__()[1:-1],
                     "chunk created at": c._creation,
                     "chunk reference count": c._reference_count,
                     "chunk references": Memory._elide_long_list(c._references)}
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

    @staticmethod
    def _elide_long_list(lst):
        lst = list(lst)
        if len(lst) <= 8:
            return lst.__repr__()[1:-1]
        else:
            return lst[:3].__repr__()[1:-1] + ", ... " + lst[-3:].__repr__()[1:-1]

    def learn(self, slots, advance=None):
        """Adds, or reinforces, a chunk in this Memory with the attributes specified by *slots*.
        The attributes, or slots, of a chunk are described using the :class:`Mapping`
        *slots*, the keys of which must be non-empty strings and are the attribute names.
        All the values of the various *slots* must be :class:`Hashable`.

        Returns the chunk created if a new chunk has been created, and ``None`` if
        instead an already existing chunk has been re-experienced and thus reinforced.

        Note that after learning one or more chunks, before :meth:`retrieve`,
        :meth:`blend` or similar methods can be called :meth:`advance` must be called,
        lest the chunk(s) learned have infinite activation.
        Because it is so common to call :meth:`advance` immediately after :meth"`learn`
        as a convenience if *advance* is not None just before :meth:`learn` returns
        :meth:`advance` with *advance* as its argument, without an argument if *advance*
        is ``True``.

        Raises a :exc:`TypeError` if an attempt is made to learn an attribute value that
        is not :class:`Hashable`. Raises a :exc:`ValueError` if no *slots* are provided,
        or if any of the keys of *slots* are not non-empty strings.

        >>> m = Memory()
        >>> m.learn({"color":"red", "size":4})
        True
        >>> m.advance()
        1
        >>> m.learn({"color":"blue", "size":4}, advance=1)
        True
        >>> m.learn({"color":"red", "size":4})
        False
        >>> m.advance()
        3
        >>> m.retrieve({"color": "red"})
        <Chunk 0000 {'color': 'red', 'size': 4} 2>
        """
        slots = self._ensure_slots(slots, True)
        signature = Memory._signature(slots, "learn")
        created = False
        if not (chunk := self.get(signature)):
            chunk = Chunk(self, slots)
            created = True
            self[signature] = chunk
            self._slot_name_index[frozenset(slots.keys())].append(chunk)
            if  self._indexed_attributes:
                self._index[Memory._signature(chunk, "learn", self._indexed_attributes)
                            ].append(chunk)
        self._cite(chunk)
        if advance is True:
            self.advance()
        elif advance is not None:
            self.advance(advance)
        return chunk if created else None

    @staticmethod
    def _ensure_slot_name(name):
        if not (isinstance(name, str) and len(name) > 0):
                raise ValueError(f"Attribute name {name} is not a non-empty string")

    @staticmethod
    def _ensure_slot_names(thing):
        if thing is None:
            return []
        if isinstance(thing, str):
            if "," in thing:
                names = [s.strip() for s in thing.split(",")]
            else:
                names = thing.split()
        else:
            names = list(thing)
        s = set()
        for n in names:
            Memory._ensure_slot_name(n)
            if n in s:
                raise ValueError(f"Duplicate attribute name {n}")
            s.add(n)
        return names

    def _ensure_slots(self, slots, learn=False):
        slots = dict(slots)
        for name in slots.keys():
            Memory._ensure_slot_name(name)
        if learn:
            for n in self._indexed_attributes:
                if slots.get(n) is None:
                    slots[n] = None
        return slots

    @staticmethod
    def _signature(slots, fname, attributes=None):
        if attributes is not None:
            result = tuple(sorted({a: slots[a] for a in attributes}.items()))
        elif not (result := tuple(sorted(slots.items()))):
            if fname:
                raise ValueError(f"No attributes provided to {fname}()")
            else:
                raise ValueError(f"No attributes provided")
        return result

    def _cite(self, chunk):
        if self._optimized_learning is None:
            if chunk._reference_count >= chunk._references.size:
                chunk._references.resize(REFERENCES_FACTOR * chunk._references.size,
                                         refcheck=False)
            chunk._references[chunk._reference_count] = self._time
        elif  chunk._reference_count < self._optimized_learning:
            if chunk._reference_count >= chunk._references.size:
                chunk._references.resize(min(REFERENCES_FACTOR * chunk._references.size,
                                             self._optimized_learning),
                                         refcheck=False)
            chunk._references[chunk._reference_count] = self._time
        elif self._optimized_learning:
            chunk._references[:-1] = chunk._references[1:]
            chunk._references[-1] = self._time
        chunk._reference_count += 1

    def forget(self, slots, when):
        """Undoes the operation of a previous call to :meth:`learn`.

        .. warning::
            Normally this method should not be used. It does not correspond to a
            biologically plausible process, and is only provided for esoteric purposes.

        The *slots* should be those supplied for the :meth:`learn` operation to be
        undone, and *when* should be the time that was current when the operation was
        performed. Returns ``True`` if it successfully undoes such an operation, and
        ``False`` otherwise.

        This method cannot be used with :attr:`optimized_learning`, and calling it when
        optimized learning is enabled raises a :exc:`RuntimeError`.
        """
        if self._optimized_learning is not None:
            raise RuntimeError("The forget() method cannot be used with optimized learning")
        slots = self._ensure_slots(slots, True)
        signature = Memory._signature(slots, "forget")
        chunk = self.get(signature)
        if not chunk:
            return False
        try:
            i = np.where(chunk._references == when)[0][0]
        except IndexError:
            return False
        if i < chunk._reference_count:
            chunk._references[i:chunk._reference_count-1] = chunk._references[i+1:chunk._reference_count]
        chunk._reference_count -= 1
        if not chunk._reference_count:
            self._slot_name_index[frozenset(chunk.keys())].remove(chunk)
            del self[signature]
            if self._indexed_attributes:
                self._index[Memory._signature(chunk, "forget", self._indexed_attributes)
                            ].remove(chunk)
        return True

    def _activations(self, conditions, extra=None, partial=True):
        slot_names = conditions.keys()
        if extra:
            slot_names = set(slot_names)
            slot_names.add(extra)
        partial_slots = []
        if partial and self._mismatch:
            exact_slots =[]
            for n, v in conditions.items():
                if s := self._similarities.get(n):
                    partial_slots.append((n, v, s))
                else:
                    exact_slots.append((n, v))
        else:
            exact_slots = list(conditions.items())
        if self._indexed_attributes and (set(a[0] for a in exact_slots)
                                         == self._indexed_attributes):
            chunks = self._index[Memory._signature(conditions,
                                                   None,
                                                   self._indexed_attributes)]
        else:
            chunks = []
            for k, candidates in self._slot_name_index.items():
                if slot_names <= k: # subset
                    for c in candidates:
                        if not all(c[n] == v for n, v in exact_slots):
                            continue
                        chunks.append(c)
        if len(chunks) == 0:
            return None, None, 0
        nchunks = len(chunks)
        with np.errstate(divide="raise", over="raise", under="ignore", invalid="raise"):
            try:
                if self._decay is not None:
                    if self._optimized_learning is None:
                        result = np.empty(nchunks)
                        for c, i in zip(chunks, count()):
                            result[i] = np.sum((self._time - c._references[0:c._reference_count])
                                               ** -self._decay)
                        result = np.log(result)
                    elif self._optimized_learning == 0:
                        counts = np.empty(nchunks)
                        ages = np.empty(nchunks)
                        for c, i in zip(chunks, count()):
                            counts[i] = c._reference_count
                            ages[i] = self._time - c._creation
                        result = (np.log(counts / (1 - self._decay))
                                  - self._decay * np.log(ages))
                    else:
                        result = np.empty(nchunks)
                        counts = ma.masked_all(nchunks)
                        ages = ma.masked_all(nchunks)
                        middles = ma.masked_all(nchunks)
                        for c, i in zip(chunks, count()):
                            if c._reference_count <= self._optimized_learning:
                                result[i] = np.sum((self._time - c._references[0:c._reference_count])
                                                   ** -self._decay)
                            else:
                                result[i] = np.sum((self._time - c._references[0:self._optimized_learning])
                                                   ** -self._decay)
                                counts[i] = c._reference_count
                                ages[i] = self._time - c._creation
                                middles[i] = c._references[0]
                        dd = 1 - self._decay
                        counts -= self._optimized_learning
                        diff = ages - middles
                        diff *= dd
                        ages **= dd
                        middles **= dd
                        tmp = ages
                        tmp -= middles
                        tmp *= counts
                        tmp /= diff
                        result = np.log(result + tmp.filled(0))
                else:
                    result = np.zeros(nchunks)
                if self._activation_history is not None:
                    initial_history_length = len(self._activation_history)
                    for c, r in zip(chunks, result):
                        self._activation_history.append({"name": c._name,
                                                         "creation_time": c._creation,
                                                         "attributes": tuple(c.items()),
                                                         "reference_count": c.reference_count,
                                                         "references": c.references,
                                                         "base_level_activation": r})
                if self._noise:
                    noise = self._rng.logistic(scale=self._noise, size=nchunks)
                    if self._fixed_noise is not None:
                        if self._fixed_noise_time != self._time:
                            self._clear_fixed_noise()
                            for c, s in zip(chunks, noise):
                                self._fixed_noise[c._name] = s
                        else:
                            for c, s, i in zip(chunks, noise, count()):
                                if x := self._fixed_noise.get(c._name):
                                    noise[i] = x
                                else:
                                    self._fixed_noise[c._name] = s
                    result += noise
                    if self._activation_history is not None:
                        for i, s in zip(count(initial_history_length), noise):
                            self._activation_history[i]["activation_noise"] = s
                if partial_slots:
                    penalties = np.empty((nchunks, len(partial_slots)))
                    for c, row in zip(chunks, count()):
                        penalties[row] = [s._similarity(c[n], v) for n, v, s in partial_slots]
                    penalties = np.sum(penalties, 1) * self._mismatch
                    result += penalties
                    if self._activation_history is not None:
                        for i, p in zip(count(initial_history_length), penalties):
                            self._activation_history[i]["mismatch"] = p
                if self._activation_history is not None:
                    for i, r in zip(count(initial_history_length), result):
                        self._activation_history[i]["activation"] = r
                        if self._threshold is not None:
                            self._activation_history[i]["meets_threshold"] = (r >= self._threshold)
                raw_activations_count = len(result)
                if self._threshold is not None:
                    m = ma.masked_less(result, self._threshold)
                    if ma.is_masked(m):
                        chunks = ma.array(chunks, mask=ma.getmask(m)).compressed()
                        result = m.compressed()
            except FloatingPointError as e:
                raise RuntimeError(f"Error when computing activations, perhaps a chunk's "
                                   f"creation or reinforcement time is not in the past? ({e})")
        if result.size == 0:
            return None, None, raw_activations_count
        else:
            return result, chunks, raw_activations_count

    def retrieve(self, slots={}, partial=False, rehearse=False):
        """Returns the chunk matching the *slots* that has the highest activation greater than or equal to this Memory's :attr:`threshold`, if any.
        If there is no such matching chunk returns ``None``.
        Normally only retrieves chunks exactly matching the *slots*; if *partial* is
        ``True`` it also retrieves those only approximately matching, using similarity
        (see :meth:`similarity`) and :attr:`mismatch` to determine closeness
        of match.

        If *rehearse* is supplied and true it also reinforces this chunk at the current
        time. No chunk is reinforced if retrieve returns ``None``.

        The returned chunk is a dictionary-like object, and its attributes can be
        extracted with Python's usual subscript notation.

        If any matching chunks were created or reinforced at or after the current time
        an :exc:`Exception` is raised.

        >>> m = Memory()
        >>> m.learn({"widget":"thromdibulator", "color":"red", "size":2})
        True
        >>> m.advance()
        1
        >>> m.learn({"widget":"snackleizer", "color":"blue", "size":1})
        True
        >>> m.advance()
        2
        >>> m.retrieve({"color":"blue"})["widget"]
        'snackleizer'
        """
        activations, chunks, ignore = self._activations(self._ensure_slots(slots),
                                                        partial=partial)
        if chunks is None:
            return None
        max = np.finfo(activations.dtype).min
        best = []
        for a, c in zip(activations, chunks):
            if a < max:
                pass
            elif a > max:
                max = a
                best = [c]
            else:
                best.append(c)
        result = random.choice(best)
        if rehearse and result:
            self._cite(result)
        return result

    def _blend(self, outcome_attribute, slots):
        Memory._ensure_slot_name(outcome_attribute)
        activations, chunks, raw = self._activations(self._ensure_slots(slots),
                                                     extra=outcome_attribute)
        if chunks is None:
            return None, None
        with np.errstate(divide="raise", over="raise", under="ignore", invalid="raise"):
            wp = np.exp(activations / self._temperature)
            wp /= np.sum(wp)
            if self._activation_history is not None:
                h = self._activation_history
                # this i malarkey is in case one or more candidates didn't clear the threshold
                i = len(h) - raw
                for p, c in zip(wp, chunks):
                    while h[i]["name"] != c._name:
                        i += 1
                        assert i < len(h)
                    h[i]["retrieval_probability"] = p
        return wp, chunks

    def blend(self, outcome_attribute, slots={}):
        """Returns a blended value for the given attribute of those chunks matching *slots*, and which contain *outcome_attribute*, and have activations greater than or equal to this Memory's threshold, if any.
        Returns ``None`` if there are no matching chunks that contain
        *outcome_attribute*. If any matching chunk has a value of *outcome_attribute*
        that is not a real number an :exc:`Exception` is raised.

        >>> m = Memory()
        >>> m.learn({"color":"red", "size":2})
        True
        >>> m.advance()
        1
        >>> m.learn({"color":"blue", "size":30})
        True
        >>> m.advance()
        2
        >>> m.learn({"color":"red", "size":1})
        True
        >>> m.advance()
        3
        >>> m.blend("size", {"color":"red"})
        1.221272238515685
        """
        probs, chunks = self._blend(outcome_attribute, slots)
        if chunks is None:
            return None
        with np.errstate(divide="raise", over="raise", under="ignore", invalid="raise"):
            try:
                return np.average(np.array([c[outcome_attribute] for c in chunks],
                                           dtype=np.float64),
                                  weights=probs)
            except Exception as e:
                raise RuntimeError(f"Error computing blended value, is perhaps the value "
                                   f"of the {outcome_attribute} slotis  not numeric in "
                                   f"one of the matching chunks? ({e})")

    def best_blend(self, outcome_attribute, iterable, select_attribute=None, minimize=False):
        """Returns two values (as a 2-tuple), describing the extreme blended value of the *outcome_attribute* over the values provided by *iterable*.
        The extreme value is normally the maximum, but can be made the minimum by setting
        *minimize* to ``True``. The *iterable* is an :class:`Iterable` of
        :class:`Mapping` objects, mapping attribute names to values, suitable for
        passing as the *slots* argument to :meth:`blend`. The first
        return value is the *iterable* value producing the best blended value, and the
        second is that blended value. If there is a tie, with two or more *iterable* values
        all producing the same, best blended value, then one of them is chosen randomly. If
        none of the values from *iterable* result in blended values of *outcome_attribute*
        then both return values are ``None``.

        This operation is particularly useful for building Instance Based Learning models.

        For the common case where *iterable* iterates over only the values of a single
        slot the *select_attribute* parameter may be used to simplify the iteration. If
        *select_attribute* is supplied and is not ``None`` then *iterable* should produce
        values of that attribute instead of :class:`Mapping` objects. Similarly the
        first return value will be the attribute value rather than a :class:`Mapping`
        object. The end of the example below demonstrates this.

        >>> m = Memory()
        >>> m.learn({"color":"red", "utility":1})
        True
        >>> m.advance()
        1
        >>> m.learn({"color":"blue", "utility":2})
        True
        >>> m.advance()
        2
        >>> m.learn({"color":"red", "utility":1.8})
        True
        >>> m.advance()
        3
        >>> m.learn({"color":"blue", "utility":0.9})
        True
        >>> m.advance()
        4
        >>> m.best_blend("utility", ({"color": c} for c in ("red", "blue")))
        ({'color': 'blue'}, 1.5149259914576285)
        >>> m.learn({"color":"blue", "utility":-1})
        True
        >>> m.advance()
        5
        >>> m.best_blend("utility", ("red", "blue"), "color")
        ('red', 1.060842632215651)
        """
        comparator = operator.gt if not minimize else operator.lt
        best_value = -math.inf if not minimize else math.inf
        best_args = []
        for thing in iterable:
            if select_attribute is not None:
                slots = { select_attribute : thing }
            else:
                slots = thing
            value = self.blend(outcome_attribute, slots)
            if value is None:
                pass
            elif value == best_value:
                best_args.append(slots)
            elif comparator(value, best_value):
                best_args = [ slots ]
                best_value = value
        old = None
        if best_args:
            result = random.choice(best_args)
            if select_attribute is not None:
                result = result[select_attribute]
            return result, best_value
        else:
            return None, None

    def discrete_blend(self, outcome_attribute, slots={}):
        """Returns the value for the given attribute of those chunks matching *slots*, that maximizes the aggregate probabilities of retrieval of those chunks.
        Also returns a second value, a dictionary  mapping the possible values
        of *outcome_attribute* to their probabilities of retrieval.
        Returns ``None`` if there are no matching chunks that contain
        *outcome_attribute*.

        >>> m = Memory()
        >>> m.learn({"kind": "tilset", "age": "old"})
        True
        >>> m.advance()
        1
        >>> m.learn({"kind": "limburger", "age": "old"})
        True
        >>> m.advance()
        2
        >>> m.learn({"kind": "tilset", "age": "old"})
        False
        >>> m.advance()
        3
        >>> m.learn({"kind": "tilset", "age": "new"})
        True
        >>> m.advance()
        4
        >>> m.discrete_blend("kind", {"age": "old"})
        ('tilset', {'tilset': 0.9540373563209859, 'limburger': 0.04596264367901423})
        """
        probs, chunks = self._blend(outcome_attribute, slots)
        if not chunks:
            return None, None
        candidates = defaultdict(list)
        for c, p in zip(chunks, probs):
            candidates[c[outcome_attribute]].append(p)
        best = []
        best_value = -math.inf
        for k, v in candidates.items():
            v = sum(v)
            candidates[k] = v
            if v > best_value:
                best = [k]
                best_value = v
            elif v == best_value:
                best.append(k)
        return (random.choice(best),
                dict(sorted(candidates.items(), key=lambda x: x[1], reverse=True)))

    def similarity(self, attributes, function=None, weight=None):
        """Assigns a similarity function and/or corresponding weight to be used when comparing attribute values with the given *attributes*.
        The *attributes* should be an :class:`Iterable` of strings, attribute names.
        The *function* should take two arguments, and return a real number between 0 and 1,
        inclusive.
        The function should be commutative; that is, if called with the same arguments
        in the reverse order, it should return the same value.
        It should also be stateless, always returning the same values if passed
        the same arguments.
        No error is raised if either of these constraints is violated, but the results
        will, in most cases, be meaningless if they are.
        If ``True`` is supplied as the *function* a default similarity function is used
        that returns one if its two arguments are ``==`` and zero otherwise.
        If only one of *function* or *weight* is supplied, it is changed without
        changing the other; the initial defaults are ``Treu`` for *function* and ``1``
        for *weight*.
        If neither *function* nor *weight* is supplied both are removed, and these
        *attributes* will no longer have an associated similarity computation, and will
        be matched only exactly.

        As a convenience, if none of the attribute names contains commas or spaces, a
        string maybe used instead of a list as the first argument to ``similarity``, the
        attribute names being separated by spaces or commas; either spaces or commas must
        be used, not a mixture. For example, both ``"decision utility"`` and
        ``"decision,utiliy"`` are equivalent to ``["decision", "utility"]``.

        An :exc:`Exception` is raised if any of the elements of *attributes* are not
        non-zero length strings, if *function* is neither :class:`callable` nor ``True``,
        of if *weight* is not a positive, real number.

        Note that for a :class:`Memory` to be successfully pickled all the similarity
        functions should be defined at top level, and be neither lambda expressions nor
        inner functions. Pickled objects include only the names of functions, and not
        their function definitions.

        >>> def f(x, y):
        ...     if y < x:
        ...         return f(y, x)
        ...     return 1 - (y - x) / y
        >>> similarity(["length", "width"], f, weight=2)

        """
        if function is not None and not (callable(function) or function is True):
            raise ValueError(f"Function {function} is neither callable nor True")
        if weight is not None and weight <= 0:
            raise ValueError(f"Similarity weight, {weight}, is not a positive number")
        for a in Memory._ensure_slot_names(attributes):
            if function is None and weight is None:
                if a in self._similarities:
                    del self._similarities[a]
            else:
                sim = self._similarities[a]
                sim._memory = self
                if function is not None and function != sim._function:
                    sim._function = function
                if weight is not None and weight != sim._weight:
                    sim._weight = weight
                sim._cache.clear()


class Chunk(dict):

    __slots__ = ["_name", "_memory", "_creation", "_references", "_reference_count" ]

    _name_counter = 0;

    def __init__(self, memory, content):
        self._name = f"{Chunk._name_counter:04d}"
        Chunk._name_counter += 1
        self._memory = memory
        self.update(content)
        self._creation = memory._time
        self._references = np.empty(1 if self._memory._optimized_learning != 0 else 0,
                                    dtype=np.int32)
        self._reference_count = 0

    def __repr__(self):
        return "<Chunk {} {} {}>".format(self._name, dict(self), self._reference_count)

    def __str__(self):
        return f"Chunk-{self._name}"

    @property
    def reference_count(self):
        """A non-negative integer, the number of times that this :class:`Chunk` has been reinforced.
        """
        return self._reference_count

    @property
    def references(self):
        """A list of real numbers, the times at which that this :class:`Chunk` has been reinforced.
        If :attr:`optimized_learning` is being used this may be just the most recent
        reinforcements, or an empty list, depending upon the value of
        :attr:`optimized_learning`
        """
        return list(self._references[:(self._reference_count
                                       if self._memory._optimized_learning is None
                                       else min(self._reference_count,
                                                self._memory._optimized_learning))])


@dataclass
class Similarity:
    _memory: Memory = None
    _function: callable = True
    _weight: float = 1.0
    _cache: lrucache = field(default_factory=lambda: lrucache(SIMILARITY_CACHE_SIZE))

    def _similarity(self, x, y):
        # returns a non-positive number that has already been weighted on a per slot basis
        if x == y:
            return 0
        if self._function is True:
            return -self._weight
        signature = (x, y)
        result = self._cache.get(signature)
        if result is not None:
            return result
        result = self._function(x, y)
        if result < self._memory._minimum_similarity:
            raise ValueError(f"similarity value, {result}, is less than the minimum "
                             f"allowed, {self._memory._minimum_similarity}")
        elif result > self._memory._maximum_similarity:
            raise ValueError(f"similarity value, {result}, is greater than the maximum "
                             f"allowed, {self._memory._maximum_similarity}")
        if not self._memory._use_actr_similarity:
            result -= 1
        result *= self._weight
        self._cache[signature] = result
        self._cache[(y, x)] = result
        return result


# Local variables:
# fill-column: 90
# End:
