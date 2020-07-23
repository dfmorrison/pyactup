# Copyright 2018-2020 Carnegie Mellon University

from pyactup import *
import pyactup

import math
import pytest
import sys

from math import isclose

def test_parameter_manipulation():
    m = Memory()
    assert m.noise == 0.25
    assert m.decay == 0.5
    assert m.temperature is None
    assert isclose(m._temperature, 0.3535534, rel_tol=0.0001)
    assert m.threshold == -10.0
    assert m.mismatch is None
    assert m.learning_time_increment == 1
    assert m.retrieval_time_increment == 0
    assert m.optimized_learning == False
    m.temperature = False
    assert m.temperature is None
    assert isclose(m._temperature, 0.3535534, rel_tol=0.0001)
    m.noise = 0.35
    m.decay = 0.6
    m.temperature = 0.7
    m.threshold = -8
    m.mismatch = 1
    m.learning_time_increment = 0
    m.retrieval_time_increment = 0.050
    assert m.noise == 0.35
    assert m.decay == 0.6
    assert m.temperature == 0.7
    assert m._temperature == 0.7
    assert m.threshold == -8
    assert m.mismatch == 1.0
    assert m.learning_time_increment == 0
    assert m.retrieval_time_increment == 0.050
    assert m.optimized_learning == False
    assert m.optimized_learning == False
    m = Memory(0.15, 0.4, 1.1, -9, 0, 2.1, 0.1, 1)
    assert m.noise == 0.15
    assert m.decay == 0.4
    assert m.temperature == 1.1
    assert m._temperature == 1.1
    assert m.threshold == -9
    assert m.mismatch == 0.0
    assert m.learning_time_increment == 2.1
    assert m.retrieval_time_increment == 0.1
    assert m.optimized_learning == True
    m.learning_time_increment = None
    m.retrieval_time_increment = None
    assert m.learning_time_increment == 0
    assert m.retrieval_time_increment == 0
    with pytest.raises(ValueError):
        m.temperature = 0
    assert m.temperature == 1.1
    assert m._temperature == 1.1
    with pytest.raises(ValueError):
        m.noise = -1
    assert m.noise == 0.15
    with pytest.raises(ValueError):
        m.decay = -0.5
    assert m.decay == 0.4
    with pytest.raises(ValueError):
        m.mismatch = -0.1
    assert m.mismatch == 0.0
    with pytest.raises(ValueError):
        m.learning_time_increment = -0.0001
    with pytest.raises(ValueError):
        m.retrieval_time_increment = -0.0001
    with pytest.warns(UserWarning):
        m = Memory(noise=0)
    m = Memory(decay=5)
    with pytest.raises(RuntimeError):
        m = Memory(decay=5, optimized_learning=True)
    m = Memory(noise=0, temperature=0.8)

def test_time():
    m = Memory()
    assert m.time == 0
    m.advance()
    assert m.time == 1
    m.advance(12.5)
    assert isclose(m.time, 13.5)
    with pytest.raises(ValueError):
        m.advance(-0.001)
    assert isclose(m.time, 13.5)
    m.reset()
    m.learn(foo=1)
    assert m.time == 1
    m.learn(foo=2, advance=2)
    assert m.time == 3
    m.learn(foo=3, advance=0)
    assert m.time == 3
    m.retrieve(advance=7, foo=3)
    assert m.time == 10
    m.retrieve(foo=3)
    assert m.time == 10
    m.blend("foo", advance=1)
    assert m.time == 11
    with pytest.raises(ValueError):
        m.learn(foo=9, advance=-0.1)
    with pytest.raises(ValueError):
        m.retrieve(foo=9, advance=-10)
    with pytest.raises(ValueError):
        m.retrieve(partial=True, foo=9, advance=-10)
    with pytest.raises(ValueError):
        m.blend("foo", advance=-11)

def test_reset():
    m = Memory(learning_time_increment=0)
    assert m.optimized_learning == False
    assert m.time == 0
    m.learn(species="African Swallow", range="400")
    assert len(m) == 1
    assert m.time == 0
    m.learn(species="European Swallow", range="300")
    assert len(m) == 2
    assert m.time == 0
    m.learn(species="African Swallow", range="400")
    assert len(m) == 2
    assert m.time == 0
    m.reset()
    m.advance(2.5)
    assert m.optimized_learning == False
    assert len(m) == 0
    assert m.time == 2.5
    m.reset(True)
    assert m.optimized_learning == True
    assert len(m) == 0
    assert m.time == 0
    m.reset()
    assert m.optimized_learning == True
    assert len(m) == 0
    assert m.time == 0
    m.learn(species="African Swallow", range="400")
    assert len(m) == 1
    assert m.time == 0
    m.learn(species="European Swallow", range="300")
    m.advance()
    assert len(m) == 2
    assert m.time == 1
    m.learn(species="African Swallow", range="400")
    m.advance()
    assert len(m) == 2
    assert m.time == 2
    m.reset(False)
    assert m.optimized_learning == False
    assert len(m) == 0
    assert m.time == 0

def test_noise():
    m = Memory()
    assert isclose(m.noise, 0.25)
    with pytest.warns(UserWarning):
        m.noise = 0
    assert m.noise == 0
    m.noise = 1
    assert isclose(m.noise, 1)
    with pytest.raises(ValueError):
        m.noise = -1

def test_temperature():
    m = Memory()
    assert m.temperature is None
    m.temperature = 1
    assert isclose(m.temperature, 1)
    m.temperature = None
    assert m.temperature is None
    m.temperature = False
    assert m.temperature is None
    with pytest.raises(ValueError):
        m.temperature = 0
    with pytest.raises(ValueError):
        m.temperature = -1
    with pytest.raises(ValueError):
        m.temperature = 0.0001
    m.temperature = 1
    m.noise = 0
    with pytest.raises(ValueError):
        m.temperature = None
    m.noise = 0.0001
    with pytest.raises(ValueError):
        m.temperature = None

def test_decay():
    m = Memory()
    assert isclose(m.decay, 0.5)
    m.decay = 0
    assert m.decay == 0
    m.decay = 1
    assert isclose(m.decay, 1)
    with pytest.raises(ValueError):
        m.decay = -1
    m.decay = 0.435
    assert isclose(m.decay, 0.435)
    m.reset(True)
    with pytest.raises(ValueError):
        m.decay = 1
    with pytest.raises(ValueError):
        m.decay = 3.14159265359
    m.reset(False)
    m.decay = 1
    with pytest.raises(RuntimeError):
        m.reset(True)
    m.decay = 2.7182818
    with pytest.raises(RuntimeError):
        m.reset(True)
    m.reset(False)
    m.temperature = 1
    m.noise = 0
    m.decay = 0
    m.learn(foo=1)
    m.advance(3)
    assert m.time == 4
    m.learn(foo=1, advance=0)
    m.advance(7)
    c = m.retrieve(foo=1)
    assert isclose(c._activation(), 0.6931471805599453)
    m.decay = None
    assert c._activation() == 0
    m.decay = 0.8
    assert isclose(c._activation(), -1.0281200094565899)
    m.reset(True)
    m.learn(foo=1, advance=0)
    m.advance(4)
    m.learning_time_increment = 0
    m.learn(foo=1)
    m.advance(7)
    c = m.retrieve(foo=1)
    assert isclose(c._activation(), 0.3842688747553493)
    m.decay = None
    assert c._activation() == 0
    m.decay = 0
    assert isclose(c._activation(), 0.6931471805599453)

def test_threshold():
    m = Memory()
    assert isclose(m.threshold, -10)
    m.threshold = None
    assert m.threshold is None
    m.threshold = -sys.float_info.max
    assert m.threshold is None
    with pytest.raises(ValueError):
        m.threshold = "string"

def test_mismatch():
    m = Memory()
    assert m.mismatch is None
    m.mismatch = 0
    assert m.mismatch == 0
    m.mismatch = 1
    assert isclose(m.mismatch, 1)
    m.mismatch = None
    assert m.mismatch is None
    m.mismatch = False
    assert m.mismatch is None
    with pytest.raises(ValueError):
        m.mismatch = -1

def test_cached_expt():
    m = Memory()
    m.learn(a=1)
    c = m.retrieve(a=1)
    for d in (0.123, 0.432, 0.897):
        m.decay = d
        assert sum(map(bool, m._expt_cache)) == 0
        assert isclose(c._cached_expt(5), math.pow(5, -d))
        assert sum(map(bool, m._expt_cache)) == 1
        for i in range(pyactup.TRANSCENDENTAL_CACHE_SIZE + 10):
            assert isclose(c._cached_expt(i + 1), math.pow(i + 1, -d))
        assert isclose(c._cached_expt(d), math.pow(d, -d))
        assert isclose(c._cached_expt(5), math.pow(5, -d))

def test_cached_ln():
    m = Memory()
    m.learn(a=1)
    m.advance()
    c = m.retrieve(a=1)
    for d in (0.123, 0.432, 0.897):
        m.decay = d
        assert sum(map(bool, m._ln_cache)) == 0
        assert isclose(c._cached_ln(7), math.log(7))
        assert sum(map(bool, m._ln_cache)) == 1
        for i in range(pyactup.TRANSCENDENTAL_CACHE_SIZE + 10):
            assert isclose(c._cached_ln(i + 1), math.log(i + 1))
        assert isclose(c._cached_ln(d), math.log(d))
        assert isclose(c._cached_ln(7), math.log(7))

def test_learn_retrieve():
    m = Memory(learning_time_increment=0)
    m.learn(a=1, b="x")
    m.learn(a=2, b="y")
    m.learn(a=3, b="z")
    m.advance()
    assert m.retrieve(a=2)["b"] == "y"
    assert m.retrieve(a=4) is None
    m.learn(a=4, b="x")
    with pytest.raises(RuntimeError):
        m.retrieve(a=4)
    m.advance()
    assert m.retrieve(a=4)["b"] == "x"
    assert isclose(sum(m.retrieve(b="x")["a"] == 4 for i in range(1000)) / 1000, 0.71, rel_tol=0.1)
    with pytest.raises(TypeError):
        m.learn(a=[1, 2])
    m.reset()
    m.learn(color="red", size=1)
    m.advance()
    m.learn(color="blue", size=1)
    m.advance()
    m.learn(color="red", size=2)
    m.advance(100)
    m.learn(color="red", size=1)
    m.advance()
    m.learn(color="red", size=1)
    m.advance()
    assert sum(m.retrieve(color="red")["size"] == 1 for i in range(100)) > 95
    m.retrieve(rehearse=True, size=2)
    m.advance()
    assert sum(m.retrieve(color="red")["size"] == 1 for i in range(100)) < 95
    m.learn(color="red", size=1)
    with pytest.raises(RuntimeError):
        m.retrieve(color="red")

def test_similarity():
    def sim(x, y):
        if y < x:
            return sim(y, x)
        return 1 - (y - x) / y
    set_similarity_function(sim, "a")
    m = Memory(mismatch=1)
    assert len(Memory._similarity_cache) == 0
    assert isclose(m._similarity(3, 3, "a"), 1)
    assert isclose(m._similarity(4, 3, "b"), 0)
    assert len(Memory._similarity_cache) == 0
    assert isclose(m._similarity(3, 4, "a"), 0.75)
    assert len(Memory._similarity_cache) == 1
    set_similarity_function(lambda x, y: sim(x, y) / 3, "a")
    assert len(Memory._similarity_cache) == 0
    assert isclose(m._similarity(4, 3, "a"), 0.25)
    assert len(Memory._similarity_cache) == 1
    assert isclose(m._similarity(3, 4, "a"), 0.25)
    assert len(Memory._similarity_cache) == 1
    assert use_actr_similarity() == False
    use_actr_similarity(True)
    assert use_actr_similarity() == True
    set_similarity_function(lambda x, y: sim(x, y) - 1, "a")
    assert len(Memory._similarity_cache) == 0
    assert isclose(m._similarity(3, 3, "a"), 1)
    assert isclose(m._similarity(4, 3, "b"), 0)
    assert isclose(m._similarity(3, 4, "a"), 0.75)
    assert isclose(m._similarity(4, 3, "a"), 0.75)

def test_retrieve_partial():
    use_actr_similarity(False)
    def sim(x, y):
        if y < x:
            return sim(y, x)
        return 1 - (y - x) / y
    def sim2(x, y):
        return sim(x, y) - 1
    set_similarity_function(sim, "a")
    m = Memory(mismatch=1, noise=0, temperature=1, learning_time_increment=0)
    m.learn(a=1, b="x")
    m.learn(a=2, b="y")
    m.learn(a=3, b="z")
    m.learn(a=4, b="x")
    m.advance()
    assert m.retrieve(a=2.9) is None
    assert m.retrieve(True, a=3.5)["b"] == "x"
    assert m.retrieve(True, a=3.1)["b"] == "z"
    assert m.retrieve(True, a=2.4)["b"] == "y"
    use_actr_similarity(True)
    set_similarity_function(sim2, "a")
    assert m.retrieve(a=2.9) is None
    assert m.retrieve(True, a=3.5)["b"] == "x"
    assert m.retrieve(True, a=3.1)["b"] == "z"
    assert m.retrieve(True, a=2.4)["b"] == "y"
    use_actr_similarity(False)

def test_blend():
    m = Memory(temperature=1, noise=0, learning_time_increment=0)
    m.learn(a=1, b=1)
    m.learn(a=2, b=2)
    m.advance()
    assert isclose(m.blend("b", a=1), 1)
    assert isclose(m.blend("b", a=2), 2)
    assert isclose(m.blend("b"), 1.5)
    m.learn(a=1, b=1)
    m.advance()
    assert isclose(m.blend("b", a=1), 1)
    assert isclose(m.blend("b", a=2), 2)
    assert isclose(m.blend("b"), 1.2928932188134525)
    m.learn(a=1, b=2)
    m.advance()
    assert isclose(m.blend("b", a=1), 1.437740775137503)
    assert isclose(m.blend("b", a=2), 2)
    assert isclose(m.blend("b"), 1.5511727705794482)
    assert isclose(m.blend("a"), 1.2017432359063303)
    m.activation_history = []
    assert isclose(m.blend("b"), 1.5511727705794482)
    c = next(d for d in m.activation_history if d["attributes"] == (("a", 1), ("b", 1)))
    assert isclose(c["retrieval_probability"], 0.4488272294205518)
    c = next(d for d in m.activation_history if d["attributes"] == (("a", 2), ("b", 2)))
    assert isclose(c["retrieval_probability"], 0.20174323590633028)
    c = next(d for d in m.activation_history if d["attributes"] == (("a", 1), ("b", 2)))
    assert isclose(c["retrieval_probability"], 0.34942953467311794)
    m.learn(a="mumble", b=1)
    m.advance()
    with pytest.raises(TypeError):
        m.blend("a", b=1)
