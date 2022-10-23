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

__version__ = "2.0.dev1"

if "dev" in __version__:
    print("PyACTUp version", __version__)

import collections.abc as abc
import csv
import io
import math
import numpy as np
import numpy.ma as ma
import operator
import pylru
import random
import sys

from collections import defaultdict
from contextlib import contextmanager
from itertools import count
from numbers import Number
from prettytable import PrettyTable
from warnings import warn

__all__ = ("Memory", "set_similarity_function", "use_actr_similarity")

DEFAULT_NOISE = 0.25
DEFAULT_DECAY = 0.5
DEFAULT_THRESHOLD = -10.0

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
    :attr:`optimized_learning`. All can be queried and set as properties on the
    ``Memory`` object. When creating a ``Memory`` object their initial values can be
    supplied as parameters.

    A ``Memory`` object can be serialized with
    `pickle <https://docs.python.org/3.6/library/pickle.html>`_, so long as any similarity
    functions it contains are defined at the top level of a module using ``def``, allowing
    Memory objects to be saved to and restored from persistent storage. Note that attempts
    to pickle a Memory object containing a similarity function defined as a lambda
    function, or as an inner function, will cause a :exc:`PicklingError` to be raised.

    If, when creating a ``Memory`` object, any of *noise*, *decay* or *mismatch* are
    negative, or if *temperature* is less than 0.01, a :exc:`ValueError` is raised.

    """

    def __init__(self,
                 noise=DEFAULT_NOISE,
                 decay=DEFAULT_DECAY,
                 temperature=None,
                 threshold=DEFAULT_THRESHOLD,
                 mismatch=None,
                 optimized_learning=False):
        self._fixed_noise = None
        self._fixed_noise_time = None
        self._temperature_param = 1 # will be reset below, but is needed for noise assignment
        self._noise = None
        self._decay = None
        self._optimized_learning = None
        self.noise = noise
        self.decay = decay
        if temperature is None and not self._validate_temperature(None, noise):
            warn(f"A noise of {noise} and temperature of None will make the temperature too low; setting temperature to 1")
            self.temperature = 1
        else:
            self.temperature = temperature
        self.threshold = threshold
        self.mismatch = mismatch
        self.optimized_learning = optimized_learning
        self._activation_history = None
        self._slot_name_index = defaultdict(list)
        # Initialize the noise RNG from the parent Python RNG, in case the latter gets seeded for determinancy.
        self._rng = np.random.default_rng([random.randint(0, MAXIMUM_RANDOM_SEED) for i in range(16)])
        self.reset()

    def __repr__(self):
        return f"<Memory {id(self)}: {len(self)}, {self._time}>"

    def reset(self, preserve_prepopulated=False):
        """Deletes the Memory's chunks and resets its time to zero.
        If *preserve_prepopulated* is false it deletes all chunks; if it is true it
        deletes all chunk references later than time zero, completely deleting those
        chunks that were created at time greater than zero.
        """
        if preserve_prepopulated:
            preserved = {k: v for k, v in self.items() if v._creation == 0}
        self.clear()
        self._slot_name_index.clear()
        self._time = 0
        if preserve_prepopulated:
            for k, v in preserved.items():
                v._references = np.empty(1, dtype=int) if self._optimized_learning else np.array([0])
                v._reference_count = 1
                self[k] = v
        self._clear_fixed_noise()

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
            self._clear_fixed_noise()
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
        The default value is ``-10``.
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
        Otherwise any defined similarity functions (see :func:`set_similarity_function`)
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

        The value of this attributed can only be changed when the :class:`Memory` object
        does not contain any chunks, typically immediately after it is created or
        :meth:`reset`. Otherwise a :exc:`RuntimeError` is raised.

        .. warning::
            Care should be taken when using optimized learning as operations such as
            ``retrieve`` that depend upon activation will not longer raise an exception if
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
    def activation_history(self):
        """A :class:`MutableSequence`, typically a :class:`list`,  into which details of the computations underlying PyACTUp operation are appended.
        If ``None``, the default, no such details are collected.
        In addition to activation computations, the resulting retrieval probabilities are
        also collected for blending operations.
        The details collected are presented as dictionaries.

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

        If this :class:`Memory` is empty, not yet containing any chunks, nothing is
        printed, and no file is created.

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

    # TODO which of these are still needed?
    _use_actr_similarity = False
    _minimum_similarity = 0
    _maximum_similarity = 1
    _similarity_functions = {}
    _similarity_cache = pylru.lrucache(SIMILARITY_CACHE_SIZE)

    # TODO possibly update docstrings for retrieve() and blend() to reflect new similarity stuff,
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

    def learn(self, slots):
        """Adds, or reinforces, a chunk in this Memory with the attributes specified by *slots*.
        The attributes, or slots, of a chunk are described using the :class:`abc.Mapping`
        *slots*, the keys of which must be non-empty strings and are the attribute names.
        All the values of the various *slots* must be :class:`Hashable`.

        Returns ``True`` if a new chunk has been created, and ``False`` if instead an
        already existing chunk has been re-experienced and thus reinforced.

        Note that after learning one or more chunks, before :meth:`retrieve` or
        :meth:`blend` or similar methods can be called :meth:`advance` must be called,
        lest the chunk(s) learned have infinite activation.

        Raises a :exc:`TypeError` if an attempt is made to learn an attribute value that
        is not :class:`Hashable`. Raises a :exc:`ValueError` if no *slots* are provided,
        or if any of the keys of *slots* are not non-empty strings.

        >>> m = Memory()
        >>> m.learn({"color":"red", "size":4})
        True
        >>> m.advance()
        1
        >>> m.learn({"color":"blue", "size":4})
        True
        >>> m.advance()
        2
        >>> m.learn({"color":"red", "size":4})
        False
        >>> m.advance()
        3
        >>> m.retrieve({"color":"red"})
        <Chunk 0000 {'color': 'red', 'size': 4} 2>
        """
        slots = Memory._ensure_slots(slots)
        signature = Memory._signature(slots, "learn")
        created = False
        if not (chunk := self.get(signature)):
            chunk = Chunk(self, slots)
            self[signature] = chunk
            self._slot_name_index[frozenset(slots.keys())].append(chunk)
            created = True
        self._cite(chunk)
        return created

    @staticmethod
    def _ensure_slot_name(name):
        if not (isinstance(name, str) and len(name) > 0):
                raise ValueError(f"Attribute name {name} is not a non-empty string")

    @staticmethod
    def _ensure_slots(slots):
        slots = dict(slots)
        for name in slots.keys():
            Memory._ensure_slot_name(name)
        return slots

    @staticmethod
    def _signature(slots, fname):
        if not (result := tuple(sorted(slots.items()))):
            raise ValueError(f"No attributes provided to {fname}()")
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
        slots = Memory._ensure_slots(slots)
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
                if f := Memory._similarity_functions.get(s):
                    partial_slots.append((n, v, f))
                else:
                    exact_slots.append((n, v))
        else:
            exact_slots = list(conditions.items())
        chunks = []
        for k, candidates in self._slot_name_index.items():
            if slot_names <= k: # subset
                for c in candidates:
                    if not all(c[n] == v for n, v in exact_slots):
                        continue
                    chunks.append(c)
        if not chunks:
            return None, None
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
                    # TODO work out if this is better than doing it all row by row; for vectorize f?
                    # TODO figure out weights and similarity caching
                    # TODO call the similarity functions earlier?
                    sims = np.empty((ncunhks, len(partial_slots)))
                    for c, row in zip(chunks, count()):
                        sims[i] = [f(c[n], v) for n, v, f in partial_slots]
                    sims = np.sum(sims, 1) * self._mismatch
                    # TODO add it in
                    # TODO include it in the activation history
                if self._activation_history is not None:
                    for i, r in zip(count(initial_history_length), result):
                        self._activation_history[i]["activation"] = r
                        if self._threshold is not None:
                            self._activation_history[i]["meets_threshold"] = (r >= self._threshold)
                if self._threshold is not None:
                    m = ma.masked_less(result, self._threshold)
                    if ma.is_masked(m):
                        chunks = ma.array(chunks, mask=ma.getmask(m)).compressed()
                        result = m.compressed()
            except FloatingPointError as e:
                raise RuntimeError(f"Error when computing activations, perhaps a chunk's "
                                   f"creation or reinforcement time is not in the past? ({e})")
        return result, chunks

    def retrieve(self, slots={}, partial=False, rehearse=False):
        """Returns the chunk matching the *slots* that has the highest activation greater than or equal to this Memory's :attr:`threshold`.
        If there is no such matching chunk returns ``None``.
        Normally only retrieves chunks exactly matching the *slots*; if *partial* is
        ``True`` it also retrieves those only approximately matching, using similarity
        (see :func:`set_similarity_function`) and :attr:`mismatch` to determine closeness
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
        activations, chunks = self._activations(Memory._ensure_slots(slots), partial=partial)
        if not chunks:
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

    def blend(self, outcome_attribute, slots=[]):
        """Returns a blended value for the given attribute of those chunks matching *slots*, and which contain *outcome_attribute*, and have activations greater than or equal to this Memory's threshold.
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
        Memory._ensure_slot_name(outcome_attribute)
        activations, chunks = self._activations(Memory._ensure_slots(slots),
                                                extra=outcome_attribute)
        if not chunks:
            return None
        with np.errstate(divide="raise", over="raise", under="ignore", invalid="raise"):
            try:
                return np.average(np.array([c[outcome_attribute] for c in chunks]),
                                  weights=np.exp(activations / self._temperature))
            except Exception as e:
                raise RuntimeError(f"Error computing blended value, is perhaps the value "
                                   f"of the {outcome_attribute} slot not numeric in one "
                                   f"of the matching chunks? ({e})")

    def best_blend(self, outcome_attribute, iterable, select_attribute=None, minimize=False):
        """Returns two values (as a 2-tuple), describing the extreme blended value of the *outcome_attribute* over the values provided by *iterable*.
        The extreme value is normally the maximum, but can be made the minimum by setting
        *minimize* to ``True``. The *iterable* is an :class:`Iterable` of
        :class:`abc.Mapping` objects, mapping attribute names to values, suitable for
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
        values of that attribute instead of :class:`abc.Mapping` objects. Similarly the
        first return value will be the attribute value rather than a :class:`abc.Mapping`
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
            value = self.blend(outcome_attribute, slots, advance=0)
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

def use_actr_similarity(value=None):
    """Whether to use "natural" similarity values, or traditional ACT-R ones.
    PyACTUp normally uses a "natural" representation of similarities, where two values
    being completely similar, identical, has a value of one; and being completely
    dissimilar has a value of zero; with various other degrees of similarity being
    positive, real numbers less than one. Traditionally ACT-R instead uses a range of
    similarities with the most dissimilar being a negative number, usually -1, and
    completely similar being zero.

    If the argument is ``False`` or ``True`` it sets the ACT-R traditional behavior
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

def set_similarity_function(function, attributes, weight=1):
    """Assigns a similarity function to be used when comparing attribute values with the given *attributes*.
    The *attributes* should be an :class:`Iterable` of strings, attribute names.
    The *function* should take two arguments, and return a real number between 0 and 1,
    inclusive.
    The function should be commutative; that is, if called with the same arguments
    in the reverse order, it should return the same value.
    It should also be stateless, always returning the same values if passed
    the same arguments.
    No error is raised if either of these constraints is violated, but the results
    will, in most cases, be meaningless if they are.

    TODO implement and document weights

    If ``True`` is supplied as the *function* a default similarity function is used that
    returns one if its two arguments are ``==`` and zero otherwise.

    >>> def f(x, y):
    ...     if y < x:
    ...         return f(y, x)
    ...     return 1 - (y - x) / y
    >>> set_similarity_function(f, ["length", "width"])
    """
    for s in attributes:
        if callable(function):
            Memory._similarity_functions[s] = function
        elif function:
            Memory._similarity_functions[s] = True
        elif s in Memory._similarity_functions:
            del Memory._similarity_functions[s]
    Memory._similarity_cache.clear()


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
                                    dtype=int)
        self._reference_count = 0

    def __repr__(self):
        return "<Chunk {} {} {}>".format(self._name, dict(self), self._reference_count)

    def __str__(self):
        return self._name

    @property
    def reference_count(self):
        """A non-negative integer, the number of times that this :class:`Chunk` has been reinforced.
        """
        return self._reference_count

    @property
    def references(self):
        """A tuple of real numbers, the times at which that this :class:`Chunk` has been reinforced.
        If :attr:`optimized_learning` is being used this may be just the most recent
        reinforcements, or an empty tuple, depending upon the value of
        :attr:`optimized_learning`
        """
        return tuple(self._references[:(self._reference_count
                                        if self._memory._optimized_learning is None
                                        else min(self._reference_count,
                                                 self._optimized_learning))])


# Local variables:
# fill-column: 90
# End:
