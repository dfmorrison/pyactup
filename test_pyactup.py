# Copyright 2018-2024 Carnegie Mellon University

import pyactup
from pyactup import Memory

import csv
import math
import numpy as np
import pickle
import pytest
import random
import sys

from math import isclose
from pprint import pp
from timeit import default_timer

def test_parameter_manipulation():
    m = Memory()
    assert m.noise == 0.25
    assert m.decay == 0.5
    assert m.temperature is None
    assert isclose(m._temperature, 0.3535534, rel_tol=0.0001)
    assert m.threshold is None
    assert m.mismatch is None
    assert m.optimized_learning == False
    m.temperature = False
    assert m.temperature is None
    assert isclose(m._temperature, 0.3535534, rel_tol=0.0001)
    m.noise = 0.35
    m.decay = 0.6
    m.temperature = 0.7
    m.threshold = -8
    m.mismatch = 1
    m.optimized_learning = 4
    assert m.noise == 0.35
    assert m.decay == 0.6
    assert m.temperature == 0.7
    assert m._temperature == 0.7
    assert m.threshold == -8
    assert m.mismatch == 1.0
    assert m.optimized_learning == 4
    m = Memory(0.15, 0.4, 1.1, -9, 0, True)
    assert m.noise == 0.15
    assert m.decay == 0.4
    assert m.temperature == 1.1
    assert m._temperature == 1.1
    assert m.threshold == -9
    assert m.mismatch == 0.0
    assert m.optimized_learning == True
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
    m.optimized_learning = True
    assert m.optimized_learning == True
    m.optimized_learning = False
    assert m.optimized_learning == False
    m.optimized_learning = 0
    assert m.optimized_learning == True
    m.optimized_learning = None
    assert m.optimized_learning == False
    m.optimized_learning = 1
    assert m.optimized_learning == 1
    m.optimized_learning = 1000
    assert m.optimized_learning == 1000
    with pytest.raises(ValueError):
        m.mismatch = -0.1
    assert m.mismatch == 0.0
    with pytest.warns(UserWarning):
        m = Memory(noise=0)
    m = Memory(decay=5)
    with pytest.raises(ValueError):
        m.optimized_learning = True
    with pytest.raises(ValueError):
        m.optimized_learning = 1
    m = Memory(optimized_learning=True)
    with pytest.raises(ValueError):
        m.decay = 1
    m = Memory(optimized_learning=4)
    with pytest.raises(ValueError):
        m.decay = 1
    with pytest.raises(ValueError):
        m.optimized_learning = 0.5
    with pytest.raises(ValueError):
        m = Memory(decay=5, optimized_learning=True)
    m = Memory()
    m.learn({"foo": "bar"})
    with pytest.raises(RuntimeError):
        m.optimized_learning = True
    with pytest.raises(RuntimeError):
        m.optimized_learning = 1
    m = Memory(optimized_learning=True)
    m.learn({"foo": "bar"})
    with pytest.raises(RuntimeError):
        m.optimized_learning = False
    m = Memory(optimized_learning=1)
    m.learn({"foo": "bar"})
    with pytest.raises(RuntimeError):
        m.optimized_learning = False
    assert m.use_actr_similarity is False
    m.use_actr_similarity = True
    assert m.use_actr_similarity is True
    m.use_actr_similarity = 0
    assert m.use_actr_similarity is False
    m.use_actr_similarity = "yup"
    assert m.use_actr_similarity is True
    assert Memory(use_actr_similarity=1).use_actr_similarity is True
    m = Memory()
    with pytest.raises(ValueError):
        m.noise = True
    with pytest.raises(ValueError):
        m.decay = True
    with pytest.raises(ValueError):
        m.temperature = True
    with pytest.raises(ValueError):
        m.mismatch = True
    with pytest.raises(ValueError):
        m.threshold = True
    with pytest.raises(ValueError):
        m = Memory(noise=True)
    with pytest.raises(ValueError):
        m = Memory(decay=True)
    with pytest.raises(ValueError):
        m = Memory(temperature=True)
    with pytest.raises(ValueError):
        m = Memory(mismatch=True)
    with pytest.raises(ValueError):
        m = Memory(threshold=True)

def test_time():
    m = Memory()
    assert m.time == 0
    m.advance()
    assert m.time == 1
    m.advance(12.5)
    assert isclose(m.time, 13.5)
    assert isclose(m.time, 13.5)
    m.reset()
    m.learn({"foo":1})
    assert m.time == 0
    m.advance()
    assert m.time == 1
    m.retrieve({"foo":3})
    assert m.time == 1
    m.blend("foo")
    assert m.time == 1
    m.learn({"foo":2})
    with pytest.raises(RuntimeError):
        m.retrieve()
    with pytest.raises(RuntimeError):
        m.blend("foo")
    m.learn({"foo":3}, advance=True)
    assert m.time == 2
    m.learn({"foo":3}, advance=17)
    assert m.time == 19
    m.advance(-1)
    assert m.time == 18
    m.advance(-1)
    assert m.time == 17
    m.learn({"foo":4}, advance=-1)
    assert m.time == 16
    m.advance(-0.01)
    assert isclose(m.time, 15.99)
    m.learn({"foo":5}, advance=-0.1)
    assert isclose(m.time, 15.89)
    with pytest.raises(Exception):
        m.advance("cheese Grommit?")

def test_reset():
    m = Memory()
    assert m.optimized_learning == False
    assert m.time == 0
    m.learn({"species":"African Swallow", "range":400})
    assert len(m) == 1
    assert m.time == 0
    m.learn({"species":"European Swallow", "range":300})
    assert len(m) == 2
    assert m.time == 0
    m.learn({"species":"African Swallow", "range":400})
    assert len(m) == 2
    assert m.time == 0
    m.reset()
    assert m.time == 0
    m.advance(2.5)
    assert m.optimized_learning == False
    assert len(m) == 0
    assert m.time == 2.5
    m.reset()
    assert m.optimized_learning == False
    assert len(m) == 0
    assert m.time == 0
    m.optimized_learning = 4
    m.reset()
    assert m.optimized_learning == 4
    assert len(m) == 0
    assert m.time == 0
    m.learn({"species":"African Swallow", "range":400})
    assert len(m) == 1
    assert m.time == 0
    m.learn({"species":"European Swallow", "range":300})
    m.advance()
    assert len(m) == 2
    assert m.time == 1
    m.learn({"species":"African Swallow", "range":400})
    m.advance()
    assert len(m) == 2
    assert m.time == 2
    m.reset()
    assert m.optimized_learning == 4
    assert len(m) == 0
    assert m.time == 0
    m.optimized_learning = False
    m.learn({"species":"African Swallow", "range":400})
    m.learn({"species":"European Swallow", "range":300})
    m.advance()
    m.learn({"species":"Python", "range":300})
    assert len(m) == 3
    assert m.time == 1
    m = Memory(index=["d"])
    m.learn({"d": "right", "a": 0.3, "u": 0.5})
    m.learn({"d": "left", "a": 0.5, "u": 0.3})
    m.learn({"d": "right", "a": 0.3, "u": 0.5})
    m.advance()
    m.learn({"d": "right", "a": 0.3, "u": 0.5})
    assert len(m) == 2
    c = m._index[(("d", "right"),)][0]
    assert c._creation == 0
    assert c._reference_count == 3
    assert list(c._references[:c._reference_count]) == [0, 0, 1]
    m.reset(True)
    c = m._index[(("d", "right"),)][0]
    assert c._creation == 0
    assert c._reference_count == 2
    assert list(c._references[:c._reference_count]) == [0, 0]

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
    m.reset()
    m.optimized_learning = True
    with pytest.raises(ValueError):
        m.decay = 1
    with pytest.raises(ValueError):
        m.decay = 3.14159265359
    m.optimized_learning = False
    m.decay = 1
    with pytest.raises(ValueError):
        m.optimized_learning = True
    m.decay = 2.7182818
    with pytest.raises(ValueError):
        m.optimized_learning = True
    m.reset()
    m.temperature = 1
    m.noise = 0
    m.decay = 0
    m.learn({"foo":1}, advance=3)
    assert m.time == 3
    m.learn({"foo":1})
    m.advance(7)
    c = m.retrieve({"foo":1})
    assert c.memory is m
    assert isclose(m._activations({})[0][0], 0.6931471805599453)
    m.decay = None
    assert isclose(m._activations({})[0][0], 0.0)
    m.decay = 0.8
    assert isclose(m._activations({})[0][0], -0.9961078949810501)
    m.reset()
    m.learn({"foo":1})
    m.advance(4)
    m.learn({"foo":1})
    m.advance(7)
    c = m.retrieve({"foo":1})
    m.decay = None
    assert isclose(m._activations({})[0][0], 0.0)
    m.decay = 0
    assert isclose(m._activations({})[0][0], 0.6931471805599453)

def test_threshold():
    m = Memory()
    assert m.threshold is None
    m.threshold = -10
    assert m.threshold -10
    m.threshold = -sys.float_info.max
    assert m.threshold == -sys.float_info.max
    with pytest.raises(ValueError):
        m.threshold = "string"
    m = Memory(temperature=1, noise=0, threshold=-3)
    for ol in [False, True, 1, 2]:
        m.reset()
        m.learn({"cheese": "tilset"})
        m.advance(100)
        assert m.retrieve() is not None
        m.advance(1000)
        assert m.retrieve() is None

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

def test_learn_retrieve():
    for m in [Memory(), Memory(index="a"), Memory(index=["b"]),
              Memory(index="a,b"), Memory(index="b a")]:
        m.learn({"a":1, "b":"x"})
        m.learn({"a":2, "b":"y"})
        m.learn({"a":3, "b":"z"})
        m.advance()
        assert m.retrieve({"a":2})["b"] == "y"
        assert m.retrieve({"a":4}) is None
        m.learn({"a":4, "b":"x"})
        with pytest.raises(RuntimeError):
            m.retrieve({"a":4})
        m.advance()
        assert m.retrieve({"a":4})["b"] == "x"
        assert isclose(sum(m.retrieve({"b":"x"})["a"] == 4 for i in range(1000)) / 1000, 0.71, rel_tol=0.1)
        with pytest.raises(TypeError):
            m.learn({"a":[1, 2]})
    for m in [Memory(), Memory(index="color"), Memory(index=["size"]),
              Memory(index=" color  , size  "), Memory(index="size color")]:
        m.learn({"color":"red", "size":1}, advance=True)
        assert m.time == 1
        m.learn({"size":1, "color":"blue"})
        m.advance()
        assert m.time == 2
        m.learn({"color":"red", "size":2})
        m.advance(100)
        m.learn({"color":"red", "size":1})
        m.advance()
        m.learn({"size":1, "color":"red"})
        m.advance()
        assert sum(m.retrieve({"color":"red"})["size"] == 1 for i in range(100)) > 95
        m.retrieve({"size":2}, rehearse=True)
        m.advance()
        assert sum(m.retrieve({"color":"red"})["size"] == 1 for i in range(100)) < 95
        m.learn({"color":"red", "size":1})
        with pytest.raises(RuntimeError):
            m.retrieve({"color":"red"})
    for m in [Memory(), Memory(index="kind"), Memory(index="kind, ripeness")]:
        m.temperature = 1
        m.noise = 0
        m.learn({"kind": "tilset", "ripeness": 9, "weight": 1.2})
        m.advance()
        m.learn({"kind": "limburger", "ripeness": 8, "weight": 0.9})
        m.advance()
        m.learn({"kind": "tilset", "ripeness": 4, "weight": 1.3})
        m.advance()
        m.learn({"ripeness": 9, "kind": "tilset", "weight": 1.2})
        m.advance()
        m.learn({"weight": 1.2, "kind": "tilset", "ripeness": 9 })
        m.advance()
        m.learn({"kind": "tilset", "weight": 1.1, "ripeness": 9 })
        m.advance()
        assert len(m.chunks) == 4
        assert m.retrieve()["kind"] == "tilset"
        assert m.retrieve()["ripeness"] == 9
        assert m.retrieve()["weight"] == 1.2

def test_similarity():
    def sim(x, y):
        if y < x:
            return sim(y, x)
        return 1 - (y - x) / y
    m = Memory(mismatch=1)
    m.similarity(["a"], sim)
    m.similarity("b", True)
    def test_one(x, y, a_len1, a_val, a_len2, b_len1, b_val, b_len2):
        a = m._similarities.get("a")
        b = m._similarities.get("b")
        c = m._similarities.get("c")
        assert c is None
        assert len(a._cache) == a_len1
        assert len(b._cache) == b_len1
        assert isclose(a._similarity(x, y), a_val)
        assert isclose(b._similarity(y, x), b_val)
        assert isclose(a._similarity(y, x), a_val)
        assert isclose(b._similarity(x, y), b_val)
        assert isclose(a._similarity(x, y), a_val)
        assert isclose(b._similarity(y, x), b_val)
        assert len(a._cache) == a_len2
        assert len(b._cache) == b_len2
    test_one(3, 3, 0, 0, 0, 0, 0, 0)
    test_one(3, 4, 0, -0.25, 2, 0, -1, 0)
    test_one(4, 3, 2, -0.25, 2, 0, -1, 0)
    m.similarity("a", weight=2)
    m.similarity(["b"], weight=10)
    test_one(4, 3, 0, -0.50, 2, 0, -10, 0)
    m.similarity(["b"])
    assert m._similarities.get("b") is None
    m.use_actr_similarity = True
    m.similarity("b d f", weight=20)
    m.similarity(["a"], lambda x, y: sim(x, y) - 1, 4)
    test_one(3, 4, 0, -1, 2, 0, -20, 0)
    with pytest.raises(ValueError):
        m.similarity("a,b,c,d,b,f,g", True)
    with pytest.raises(ValueError):
        m.similarity("a,b,c,d,b,f,g")

def test_retrieve_partial():
    def sim(x, y):
        if y < x:
            return sim(y, x)
        return 1 - (y - x) / y
    def sim2(x, y):
        return sim(x, y) - 1
    for m in [Memory(mismatch=1, noise=0, temperature=1),
              Memory(mismatch=1, noise=0, temperature=1, index="b"),
              Memory(index="a,b", mismatch=1, noise=0, temperature=1)]:
        m.similarity(["a"], sim)
        m.learn({"a":1, "b":"x"})
        m.learn({"a":2, "b":"y"})
        m.learn({"a":3, "b":"z"})
        m.learn({"a":4, "b":"x"})
        m.advance()
        assert m.retrieve({"a":2.9}) is None
        assert m.retrieve({"a":3.5}, True)["b"] == "x"
        assert m.retrieve({"a":3.1}, True)["b"] == "z"
        assert m.retrieve({"a":2.4}, True)["b"] == "y"
        m.use_actr_similarity = True
        m.similarity(["a"], sim2)
        assert m.retrieve({"a":2.9}) is None
        assert m.retrieve({"a":3.5}, True)["b"] == "x"
        assert m.retrieve({"a":3.1}, True)["b"] == "z"
        assert m.retrieve({"a":2.4}, True)["b"] == "y"
    m = Memory(mismatch=0, noise=0, temperature=1)
    m.similarity(["a"], sim)
    m.learn({"a":1, "b":"x"})
    m.learn({"a":2, "b":"y"})
    m.learn({"a":3, "b":"z"})
    m.learn({"a":4, "b":"x"})
    m.advance()
    many = map(lambda x: x["b"], [m.retrieve({"a":1.1}, True) for i in range(500)])
    assert "x" in many
    assert "y" in many
    assert "z" in many

def test_extra_activation():
    m = Memory(temperature=1, noise=0)
    m.learn({"a":1, "b":1})
    m.advance()
    m.learn({"a":2, "b":2})
    m.advance()
    assert m.extra_activation is None
    assert m.retrieve()["a"] == 2
    m.extra_activation = lambda c: 1 if c["b"] == 1 else 0
    assert isinstance(m.extra_activation, tuple) and len(m.extra_activation) == 1
    assert m.retrieve()["a"] == 1
    m.extra_activation = (lambda c: 1 if c["b"] == 1 else 0,
                          lambda c: -1 if c["a"] == 1 else 0)
    assert isinstance(m.extra_activation, tuple) and len(m.extra_activation) == 2
    assert m.retrieve()["a"] == 2
    m.extra_activation = [lambda c: 1 if c["b"] == 1 else 0,
                          lambda c: 1 if c["b"] != 1 else None,
                          lambda c: -1 if c["a"] == 1 else 0]
    assert isinstance(m.extra_activation, tuple) and len(m.extra_activation) == 3
    with pytest.raises(RuntimeError):
        m.retrieve()
    m.extra_activation = None
    assert m.extra_activation is None
    m.extra_activation = ()
    assert m.extra_activation is None
    m.extra_activation = []
    assert m.extra_activation is None
    m.extra_activation = False
    assert m.extra_activation is None
    assert m.retrieve()["a"] == 2
    with pytest.raises(ValueError):
        m.extra_activation = True
    with pytest.raises(ValueError):
        m.extra_activation = 1
    with pytest.raises(ValueError):
        m.extra_activation = (1,)

def test_activation_history():
    m = Memory(temperature=1, noise=0)
    m.learn({"a":1, "b":1})
    m.learn({"a":2, "b":2})
    m.advance()
    assert m.activation_history is None
    m.activation_history = True
    assert m.activation_history == []
    m.activation_history == []
    m.retrieve({"a":2})
    assert len(m.activation_history) == 1
    assert isclose(m.activation_history[0]["time"], 1)
    assert isclose(m.activation_history[0]["base_level_activation"], 0)
    assert isclose(m.activation_history[0]["activation"], 0)
    m.advance(10)
    m.extra_activation = lambda c: -1
    m.retrieve({"a":2})
    assert len(m.activation_history) == 2
    assert isclose(m.activation_history[0]["time"], 1)
    assert isclose(m.activation_history[0]["base_level_activation"], 0)
    assert isclose(m.activation_history[0]["activation"], 0)
    assert isclose(m.activation_history[1]["time"], 11)
    assert isclose(m.activation_history[1]["base_level_activation"], -1.1989476363991853)
    assert isclose(m.activation_history[1]["extra_activation"], -1)
    assert isclose(m.activation_history[1]["activation"], -2.1989476363991853)
    def setup_partial(m, fn):
        m.similarity(["x", "y"], fn)
        m.learn({"w": 0, "x": 0, "y": 0, "z": 0})
        m.advance()
        m.learn({"w": 1, "x": 0.5, "y": 0.3, "z": 0})
        m.advance()
        m.learn({"w": 0, "x": 0.2, "y": 0.1, "z": 0})
        m.advance()
        m.learn({"w": 1, "x": 0.7, "y": 0.2, "z": 0})
        m.advance()
        m.learn({"w": 100, "x": 0, "y": 0, "z": 1})
        m.advance()
        m.learn({"w": 100, "x": 0.5, "y": 0.3, "z": 1})
        m.activation_history = True
        m.blend("w", {"x": 0.05, "y": 0.05, "z": 0})
    m = Memory(mismatch=1)
    setup_partial(m, lambda x, y: 1 - abs(x - y))
    assert isclose(m.activation_history[0]["similarities"]["x"], 0.95)
    assert isclose(m.activation_history[0]["similarities"]["y"], 0.95)
    assert isclose(m.activation_history[3]["similarities"]["x"], 0.35)
    assert isclose(m.activation_history[3]["similarities"]["y"], 0.85)
    m = Memory(mismatch=1, use_actr_similarity=True)
    setup_partial(m, lambda x, y: -abs(x - y))
    assert isclose(m.activation_history[0]["similarities"]["x"], -0.05)
    assert isclose(m.activation_history[0]["similarities"]["y"], -0.05)
    assert isclose(m.activation_history[3]["similarities"]["x"], -0.65)
    assert isclose(m.activation_history[3]["similarities"]["y"], -0.15)

def test_noise_distribution():
    m = Memory(temperature=1, noise=0)
    assert m.noise_distribution is None
    m.learn({"x": 4, "y": 17})
    m.advance(10)
    m.activation_history = True
    m.retrieve({"x": 4})
    assert isclose(m.activation_history[0]["activation"], -1.1512925464970227)
    m.noise_distribution = lambda: -2
    assert m.noise_distribution is not None
    m.activation_history = True
    m.retrieve({"x": 4})
    assert isclose(m.activation_history[0]["activation"], -1.1512925464970227)
    m.noise = 0.25
    m.activation_history = True
    m.retrieve({"x": 4})
    assert isclose(m.activation_history[0]["activation"], -1.1512925464970227 - 0.5)
    m.noise = 0.5
    m.activation_history = True
    m.retrieve({"x": 4})
    assert isclose(m.activation_history[0]["activation"], -1.1512925464970227 - 1)
    m.noise_distribution = None
    assert m.noise_distribution is None
    with pytest.raises(ValueError):
        m.noise_distribution = ""

def test_blend():
    for m in [Memory(temperature=1, noise=0),
              Memory(temperature=1, noise=0, index="a"),
              Memory(temperature=1, noise=0, index="b"),
              Memory(temperature=1, noise=0, index="a,b")]:
        m.learn({"a":1, "b":1})
        m.learn({"a":2, "b":2})
        m.advance()
        assert isclose(m.blend("b", {"a":1}), 1)
        assert isclose(m.blend("b", {"a":2}), 2)
        assert isclose(m.blend("b"), 1.5)
        m.learn({"a":1, "b":1})
        m.advance()
        assert isclose(m.blend("b", {"a":1}), 1)
        assert isclose(m.blend("b", {"a":2}), 2)
        assert isclose(m.blend("b"), 1.2928932188134525)
        m.learn({"a":1, "b":2})
        m.advance()
        assert isclose(m.blend("b", {"a":1}), 1.437740775137503)
        assert isclose(m.blend("b", {"a":2}), 2)
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
        m.learn({"a":"mumble","b":1})
        m.advance()
        with pytest.raises(TypeError):
            m.blend("a", b=1)
    for m in [Memory(temperature=1, noise=0),
              Memory(temperature=1, noise=0, index="b"),
              Memory(temperature=1, noise=0, index="e"),
              Memory(temperature=1, noise=0, index="n"),
              Memory(temperature=1, noise=0, index="s"),
              Memory(temperature=1, noise=0, index="b e"),
              Memory(temperature=1, noise=0, index="b e n"),
              Memory(temperature=1, noise=0, index="b e n   s")]:
        m.similarity(["n"], lambda x, y: 1 - abs(x - y) / 100, 0.5)
        s_sims = {("a", "b"): 0.5, ("a", "c"): 0.1, ("b", "c"): 0.9}
        m.similarity(["s"], lambda x, y: s_sims[tuple(sorted([x, y]))], 0.75)
        m.mismatch = 1
        m.learn({"b": 0, "e": 0, "n": 0, "s": "a"}, advance=True)
        m.learn({"b": 100, "e": 0, "n": 50, "s": "b"}, advance=True)
        m.learn({"b": 0, "e": 1, "n": 0, "s": "a"}, advance=True)
        m.learn({"b": 0, "e": 0, "n": 0, "s": "c"}, advance=True)
        m.learn({"b": -100, "e": 0, "n": 10, "s": "c"}, advance=1000)
        m.learn({"b": 50, "e": 0, "n": 90, "s": "a"}, advance=True)
        m.learn({"b": 100, "e": 0, "n": 50, "s": "b"}, advance=True)
        m.learn({"b": 10, "e": 0, "n": 50, "s": "b"}, advance=True)
        m.learn({"b": 100, "e": 0, "n": 50, "s": "b"}, advance=True)
        m.learn({"b": 0, "e": 1, "n": 0, "s": "a"}, advance=True)
        m.activation_history = True
        bv = m.blend("b", {"e": 0, "n": 35, "s": "b"})
        assert isclose(bv, 65.11022723666353)
        assert len(m.activation_history) == 6
        for i, c, b, mm, a, p in [
                (0, 0, -3.4583575101768043, -0.55, -4.008357510176804, 0.009142172839621356),
                (1, 1, 0.2139847941319404, -0.07500000000000001, 0.1389847941319404, 0.5783853778285616),
                (2, 3, -3.4568686753298423, -0.24999999999999997, -3.7068686753298423, 0.012359029390679085),
                (3, 4, -3.456371410246588, -0.19999999999999998, -3.6563714102465883, 0.012999152795579281),
                (4, 1004, -0.8047189562170503, -0.65, -1.4547189562170502, 0.11751155154774308),
                (5, 1006, -0.5493061443340549, -0.07500000000000001, -0.624306144334055, 0.2696027155978155)]:
            h = m.activation_history[i]
            assert h["creation_time"] == c
            assert isclose(h["base_level_activation"], b)
            assert isclose(h["mismatch"], mm)
            assert isclose(h["activation"], a)
            assert isclose(h["retrieval_probability"], p)
        m.threshold = -3
        m.activation_history = True
        bv = m.blend("b", {"e": 0, "n": 35, "s": "b"})
        assert isclose(bv, 68.78318688351413)
        assert len(m.activation_history) == 6
        for i, c, b, mm, a, p in [
                (0, 0, -3.4583575101768043, -0.55, -4.008357510176804, None),
                (1, 1, 0.2139847941319404, -0.07500000000000001, 0.1389847941319404, 0.5990529161137754),
                (2, 3, -3.4568686753298423, -0.24999999999999997, -3.7068686753298423, None),
                (3, 4, -3.456371410246588, -0.19999999999999998, -3.6563714102465883, None),
                (4, 1004, -0.8047189562170503, -0.65, -1.4547189562170502, 0.1217106108318589),
                (5, 1006, -0.5493061443340549, -0.07500000000000001, -0.624306144334055, 0.27923647305436566)]:
            h = m.activation_history[i]
            assert h["creation_time"] == c
            assert isclose(h["base_level_activation"], b)
            assert isclose(h["mismatch"], mm)
            assert isclose(h["activation"], a)
            if p:
                assert h["meets_threshold"]
                assert isclose(h["retrieval_probability"], p)
            else:
                assert not h["meets_threshold"]
                assert h.get("retrieval_probability") is None

def test_best_blend():
    for m in [Memory(temperature=1, noise=0),
              Memory(temperature=1, noise=0, index="x"),
              Memory(temperature=1, noise=0, index="y"),
              Memory(temperature=1, noise=0, index="u"),
              Memory(temperature=1, noise=0, index="x y"),
              Memory(temperature=1, noise=0, index="u x y")]:
        m.learn({"u":0, "x":"a","y":1})
        m.advance()
        m.learn({"u":1, "x":"b","y":2})
        m.advance()
        m.learn({"u":-1, "x":"a","y":2})
        m.advance()
        m.learn({"u":0.5, "x":"b","y":1})
        m.advance()
        m.learn({"u":1, "x":"a", "y":1})
        m.advance()
        m.learn({"u":-0.2, "x":"b", "y":1})
        m.advance()
        a, v = m.best_blend("u", ({"x": x} for x in "ab"))
        assert a["x"] == "b"
        assert isclose(v, 0.26469341839060034)
        a, v = m.best_blend("u", ("a", "b"), "x")
        assert a == "b"
        assert isclose(v, 0.26469341839060034)
        a, v = m.best_blend("u", ({"x": x, "y": y} for x in "ab" for y in range(1, 3)))
        assert a["x"] == "b"
        assert a["y"] == 2
        assert isclose(v, 1.0)
        a, v = m.best_blend("u", ({"x": x, "y": y} for x in "ab" for y in range(1, 4)))
        assert a["x"] == "b"
        assert a["y"] == 2
        assert isclose(v, 1.0)
        a, v = m.best_blend("u", ({"x": x, "y": y} for x in "ab" for y in range(1, 2)))
        assert a["x"] == "a"
        assert a["y"] == 1
        assert isclose(v, 0.6339745962155614)
        a, v = m.best_blend("u", "ab", select_attribute="x", minimize=True)
        assert a == "a"
        assert isclose(v, 0.128211304635919)
    for m in [Memory(temperature=0.35, noise=0.25),
              Memory(temperature=0.35, noise=0.25, index="x"),
              Memory(temperature=0.35, noise=0.25, index="u"),
              Memory(temperature=0.35, noise=0.25, index="x y")]:
        m.learn({"u":0, "x":"a"})
        m.advance()
        m.learn({"u":1, "x":"b"})
        m.advance()
        m.learn({"u":-1, "x":"a"})
        m.advance()
        m.learn({"u":0.5, "x":"b"})
        m.advance()
        m.learn({"u":1, "x":"a"})
        m.advance()
        m.learn({"u":-0.2, "x":"b"})
        m.advance()
        assert 500 < sum(m.best_blend("u", ({"x": x} for x in "ab"))[0]["x"] == "a" for i in range(1000)) < 800
        assert m.time == 6
        m.best_blend("u", ({"x": x} for x in "ab"))
        assert m.time == 6
        m.advance()
        m.best_blend("u", "ab", "x")
        assert m.time == 7
        m.advance()
        m.best_blend("u", ({"x": x} for x in "ab"))
        assert m.time == 8
        m.best_blend("u", "ab", select_attribute="x")
        assert m.time == 8
        m.learn({"u":"not a number", "x":"a"})
        m.advance()
        assert m.time == 9
        with pytest.raises(Exception):
            m.best_blend("u", "ab", "x")
        assert m.time == 9
        a, v = m.best_blend("u", ({"x": x} for x in "bc"))
        assert a["x"] == "b"
        a, v = m.best_blend("u", ({"x": x} for x in "cde"))
        assert a is None
        assert v is None

def test_discrete_blend():
    for m in [Memory(temperature=1, noise=0),
              Memory(temperature=1, noise=0, index="s"),
              Memory(temperature=1, noise=0, index="o"),
              Memory(temperature=1, noise=0, index="s o")]:
        for _ in range(10):
            for i in range(10):
                for j in [0, 1, 2, 3, 3, 4, 5, 6, 7, 5, 8, 5, 9]:
                    m.learn({"o":j, "s":i})
                    m.advance()
        b, p = m.discrete_blend("o", {"s": 5})
        assert b == 5
        assert isclose(p[3], 0.1526351150445461)
        assert isclose(p[7], 0.07737041353204682)
        b, p = m.discrete_blend("o")
        assert b == 5
        assert isclose(p[4], 0.07519141419785559)

def test_mixed_slots():
    def run_once(d_ret_u=0, d_ret_a=0, d_ret_m=None,
                 c_ret_u=0, c_ret_a=0, c_ret_m=None,
                 d_blnd_u=0, d_blnd_a=0, d_blnd_m=None,
                 c_blnd_u=0, c_blnd_a=0, c_blnd_m=None,
                 s_blnd_u=0, s_blnd_a=0, s_blnd_m=None,
                 d_best=None, d_best_v=0, d_best_a=0, d_best_m=None,
                 c_best=None, c_best_v=0, c_best_a=0, c_best_m=None,
                 print_only=False): # print_only=True useful for debugging, etc.

        m.reset()
        m.learn({"decision":"A", "color":"red", "size":1, "utility":0})
        m.advance()
        m.learn({"decision":"A", "color":"blue", "size":4, "utility":100})
        m.advance()
        m.learn({"decision":"A", "color":"red", "size":3, "utility":10})
        m.advance()
        m.learn({"decision":"B", "color":"red", "size":3, "utility":50})
        m.advance()

        ah = []
        m.activation_history = ah
        r = m.retrieve({"decision":"A"}, partial=True)
        mp = ah[-1].get("mismatch")
        if print_only:
            print("d_ret_u =", r["utility"], ", d_ret_a =", ah[-1]["activation"], ", d_ret_m =", mp)
        else:
            assert isclose(r["utility"], d_ret_u)
            assert isclose(ah[-1]["activation"], d_ret_a)
            assert isclose(mp, d_ret_m) if d_ret_m is not None else not mp

        ah.clear()
        r = m.retrieve({"color":"red"}, partial=True)
        mp = ah[-1].get("mismatch")
        if print_only:
            print("c_ret_u =", r["utility"], ", c_ret_a =", ah[-1]["activation"],  ", c_ret_m = ", mp)
        else:
            assert isclose(r["utility"], c_ret_u)
            assert isclose(ah[-1]["activation"], c_ret_a)
            assert mp is c_ret_m or isclose(mp, c_ret_m)

        ah.clear()
        b = m.blend("utility", {"decision":"A"})
        mp = ah[-1].get("mismatch")
        if print_only:
            print("d_blnd_u =", b, ", d_blnd_a =", ah[-1]["activation"], ", d_blnd_m =", mp)
        else:
            assert isclose(b, d_blnd_u)
            assert isclose(ah[-1]["activation"], d_blnd_a)
            assert (mp is d_blnd_m or isclose(mp, d_blnd_m)) if d_blnd_m is not None else not mp

        ah.clear()
        b = m.blend("utility", {"color":"red"})
        mp = ah[-1].get("mismatch")
        if print_only:
            print("c_blnd_u =", b, ", c_blnd_a =", ah[-1]["activation"], ", c_blnd_m =", mp)
        else:
            assert isclose(b, c_blnd_u)
            assert isclose(ah[-1]["activation"], c_blnd_a)
            assert mp is c_blnd_m or isclose(mp, c_blnd_m)

        ah.clear()
        b = m.blend("utility", {"size":2})
        mp = ah[-1].get("mismatch") if ah else None
        if print_only:
            print("s_blnd_u =", b, ", s_blnd_a =", ah and ah[-1]["activation"], ", s_blnd_m =", mp)
        else:
            assert isclose(b, s_blnd_u) if s_blnd_u is not None else not b
            assert isclose(ah[-1]["activation"], s_blnd_a) if ah else s_blnd_a is None
            assert mp is s_blnd_m or isclose(mp, s_blnd_m)

        ah.clear()
        d, v = m.best_blend("utility", "AB", "decision")
        mp = ah[-1].get("mismatch")
        if print_only:
            print("d_best =", d, ", d_best_v =", v, ", d_best_a =", ah[-1]["activation"], ", d_best_m =", mp)
        else:
            assert d == d_best
            assert v == d_best_v
            assert isclose(ah[-1]["activation"], d_best_a)
            assert mp is d_best_m or isclose(mp, d_best_m)

        ah.clear()
        c, v = m.best_blend("utility", ("red", "blue"), "color")
        mp = ah[-1].get("mismatch")
        if print_only:
            print("c_best =", d, ", c_best_v =", v, ", c_best_a =", ah[-1]["activation"], ", c_best_m =", mp)
        else:
            assert d == c_best
            assert isclose(v, c_best_v)
            assert isclose(ah[-1]["activation"], c_best_a)
            assert mp is c_best_m or isclose(mp, c_best_m)

    for m in [Memory(temperature=1, noise=0),
              Memory(temperature=1, noise=0, index="decision"),
              Memory(temperature=1, noise=0, index="decision color"),
              Memory(temperature=1, noise=0, index="decision color size"),
              Memory(temperature=1, noise=0, index="color"),
              Memory(temperature=1, noise=0, index="size"),
              Memory(temperature=1, noise=0, index="color size"),
              Memory(temperature=1, noise=0, index="utility"),
              Memory(temperature=1, noise=0, index="decision size color utility")]:
        run_once(10, -0.3465735902799726, None,
                 50, 0, None,
                 36.31698208548453, -0.3465735902799726, None,
                 25.85786437626905, 0, None,
                 None, None, None,
                 "B", 50, 0, None,
                 "B", 100, -0.5493061443340549, None)
        m.mismatch = 1
        run_once(10, -0.3465735902799726, None,
                 50, 0, None,
                 36.31698208548453, -0.3465735902799726, None,
                 25.85786437626905, 0, None,
                 None, None, None,
                 "B", 50, 0, None,
                 "B", 100, -0.5493061443340549, None)
        m.similarity(["color"], True)
        run_once(10, -0.3465735902799726, None,
                 50, 0, 0,
                 36.31698208548453, -0.3465735902799726, None,
                 32.366410445083744, 0, 0,
                 None, None, None,
                 "B", 50, 0, None,
                 "B", 56.669062843109664, -1, -1)
        m.similarity(["size"], lambda x, y: 1 - abs(x - y) / 4)
        run_once(10, -0.3465735902799726, None,
                 50, 0, 0,
                 36.31698208548453, -0.3465735902799726, None,
                 32.366410445083744, 0, 0,
                 38.406038686568394, -0.25, -0.25,
                 "B", 50, 0, None,
                 "B", 56.669062843109664, -1, -1)

def test_fixed_noise():
    N = 300
    for m in [Memory(), Memory(index="n")]:
        for i in range(N):
            m.learn({"n":i})
            m.advance()
        ah = []
        m.activation_history = ah
        m.retrieve()
        m.retrieve()
        m.retrieve()
        for i in range(N):
            assert ah[i]["activation_noise"] != ah[i + N]["activation_noise"]
            assert ah[i]["activation_noise"] != ah[i + 2 * N]["activation_noise"]
            assert ah[i + N]["activation_noise"] != ah[i + 2 * N]["activation_noise"]
        ah.clear()
        with m.fixed_noise:
            m.retrieve()
            m.retrieve()
            m.retrieve()
        for i in range(N):
            assert ah[i]["activation_noise"] == ah[i + N]["activation_noise"]
            assert ah[i]["activation_noise"] == ah[i + 2 * N]["activation_noise"]
            assert ah[i + N]["activation_noise"] == ah[i + 2 * N]["activation_noise"]
        ah.clear()
        with m.fixed_noise:
            m.retrieve()
            m.advance()
            m.retrieve()
            m.retrieve()
        for i in range(N):
            assert ah[i]["activation_noise"] != ah[i + N]["activation_noise"]
            assert ah[i]["activation_noise"] != ah[i + 2 * N]["activation_noise"]
            assert ah[i + N]["activation_noise"] == ah[i + 2 * N]["activation_noise"]

def test_forget():
    for m in [Memory(), Memory(index="n"), Memory(index="s"), Memory(index="n s")]:
        assert not m.forget({"n":1}, 0)
        m.learn({"n":1})
        m.advance()
        assert not m.forget({"n":1}, 1)
        assert len(m) == 1
        assert m.forget({"n":1}, 0)
        assert len(m) == 0
        m.learn({"n":1, "s":"foo"})
        m.advance()
        m.learn({"n":2, "s":"bar"})
        m.advance()
        m.learn({"n":1, "s":"foo"})
        m.advance()
        assert len(m) == 2
        assert m.forget({"n":1, "s":"foo"}, 1)
        assert len(m) == 2
        assert m.forget({"s":"bar", "n":2}, 2)
        assert len(m) == 1
        assert m.chunks[0].references == (3,)
        for ol in [True, 1, 2, 1000]:
            m.reset()
            m.optimized_learning = ol
            m.learn({"n":1})
            m.advance()
            with pytest.raises(RuntimeError):
                m.forget({"n": 1}, 0)

def test_chunks_and_references():
    # We're depending upon chunks being in initial insertion order here; is that really
    # part of our contract, or is it just an unsupported artifact of how dicts now work?
    for m in [Memory(), Memory(index="n")]:
        assert len(m.chunks) == 0
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 1
        assert m.chunks[0].reference_count == 1
        assert m.chunks[0].references == (0,)
        m.learn({"n":2})
        m.advance()
        assert len(m.chunks) == 2
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 2
        assert m.chunks[0].reference_count == 2
        assert m.chunks[0].references == (0, 2)
        assert m.chunks[1].reference_count == 1
        assert m.chunks[1].references == (1,)
        m.reset()
        m.optimized_learning = True
        assert len(m.chunks) == 0
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 1
        assert m.chunks[0].references == ()
        m.learn({"n":2})
        m.advance()
        assert len(m.chunks) == 2
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 2
        assert m.chunks[0].reference_count == 2
        assert m.chunks[0].references == ()
        assert m.chunks[1].reference_count == 1
        assert m.chunks[1].references == ()
        m.reset()
        m.optimized_learning = 1
        assert len(m.chunks) == 0
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 1
        assert m.chunks[0].references == (0,)
        m.learn({"n":2})
        m.advance()
        assert len(m.chunks) == 2
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 2
        assert m.chunks[0].reference_count == 2
        assert m.chunks[0].references == (2,)
        assert m.chunks[1].reference_count == 1
        assert m.chunks[1].references == (1,)
        def f(ol):
            m.reset()
            m.optimized_learning = ol
            m.learn({"a1":1, "a2":2, "a3":3})
            m.learn({"a2":2, "a1":1, "a3":3})
            m.advance()
            m.learn({"a3":3, "a1":1, "a2":2})
            m.advance()
            m.learn({"a3":3, "a1":1, "a2":20})
            m.advance()
            m.learn({"a3":3, "a2":2, "a1":1})
            m.learn({"a1":10, "a3":3, "a2":2})
            m.advance()
            m.learn({"a1":1, "a3":3, "a2":2})
            m.learn({"a1":1, "a3":3, "a2":2})
            m.advance()
            m.learn({"a2":2, "a3":3, "a1":1})
            m.advance()
            m.learn({"a2":2, "a1":1, "a3":3})
            m.advance()
            m.learn({"a1":1, "a3":3, "a2":2})
            m.advance()
            m.learn({"a3":3, "a1":1, "a2":20})
            assert len(m.chunks) == 3
            assert m.chunks[0].reference_count == 9
            assert m.chunks[1].reference_count == 2
            assert m.chunks[2].reference_count == 1
        f(False)
        assert m.chunks[2].references == (3,)
        assert m.chunks[0].references == (0, 0, 1, 3, 4, 4, 5, 6, 7)
        assert m.chunks[1].references == (2, 8)
        f(True)
        assert m.chunks[0].references == ()
        assert m.chunks[1].references == ()
        assert m.chunks[2].references == ()
        f(5)
        assert m.chunks[0].references == (4, 4, 5, 6, 7)
        assert m.chunks[1].references == (2, 8)
        assert m.chunks[2].references == (3,)
        f(4)
        assert m.chunks[0].references == (4, 5, 6, 7)
        assert m.chunks[1].references == (2, 8)
        assert m.chunks[2].references == (3,)
        f(2)
        assert m.chunks[0].references == (6, 7)
        assert m.chunks[1].references == (2, 8)
        assert m.chunks[2].references == (3,)
        f(1)
        assert m.chunks[0].references == (7,)
        assert m.chunks[1].references == (8,)
        assert m.chunks[2].references == (3,)

def pickle_sim_1(x, y):
    return 1 - abs(x - y) / 100

def pickle_sim_2(x, y):
    return {("a", "b"): 0.5, ("a", "c"): 0.1, ("b", "c"): 0.9}[tuple(sorted([x, y]))]

def test_pickle():
    def capture():
        m.activation_history = True
        r = m.retrieve({"e": 1})
        bv = m.blend("b", {"e": 0, "n": 35, "s": "b"})
        return [len(m),
                m.time,
                m.chunks,
                m.noise or r,
                m.noise or bv,
                m.noise or m.activation_history,
                m.noise,
                m.decay,
                m.temperature,
                m.threshold,
                m.mismatch,
                m.optimized_learning,
                m._indexed_attributes,
                m._index,
                m._slot_name_index]
    for m in [Memory(temperature=0.97, noise=0, decay=0.43, threshold=-2.9, mismatch=1.1,
                     optimized_learning=2),
              Memory(temperature=0.97, noise=0, decay=0.43, threshold=-2.9, mismatch=1.1,
                     optimized_learning=2, index="b"),
              Memory(temperature=0.97, noise=0, decay=0.43, threshold=-2.9, mismatch=1.1,
                     optimized_learning=2, index="b e"),
              Memory(temperature=0.97, noise=0, decay=0.43, threshold=-2.9, mismatch=1.1,
                     optimized_learning=2, index="b e n s")]:
        m.similarity(["n"], pickle_sim_1 , 0.5)
        m.similarity(["s"], pickle_sim_2 , 0.75)
        m.learn({"b": 0, "e": 0, "n": 0, "s": "a"}, advance=True)
        m.learn({"b": 100, "e": 0, "n": 50, "s": "b"}, advance=True)
        m.learn({"b": 0, "e": 1, "n": 0, "s": "a"}, advance=True)
        m.learn({"b": 0, "e": 0, "n": 0, "s": "c"}, advance=True)
        m.learn({"b": -100, "e": 0, "n": 10, "s": "c"}, advance=1000)
        m.learn({"b": 50, "e": 0, "n": 90, "s": "a"}, advance=True)
        m.learn({"b": 100, "e": 0, "n": 50, "s": "b"}, advance=True)
        m.learn({"b": 10, "e": 0, "n": 50, "s": "b"}, advance=True)
        m.learn({"b": 100, "e": 0, "n": 50, "s": "b"}, advance=True)
        m.learn({"b": 0, "e": 1, "n": 0, "s": "a"}, advance=True)
        save = capture()
        sys.setrecursionlimit(100_000)
        m = pickle.loads(pickle.dumps(m))
        assert capture() == save
        m.noise=0.273
        m = pickle.loads(pickle.dumps(m))
        assert capture() != save
        save = capture()
        m = pickle.loads(pickle.dumps(m))
        assert capture() == save
        m.similarity(["n"], lambda x, y: 1 - abs(x - y) / 100, 0.5)
        with pytest.raises(Exception):
            pickle.dumps(m)

def test_index():
    m = Memory(index="a b")
    assert set(m.index) == {"a", "b"}
    c = m.learn({"a": 0})
    assert len(c) == 2 and c["b"] is None
    m = Memory()
    c = m.learn({"a": 0})
    assert len(c) == 1 and c.get("b") is None
    with pytest.raises(KeyError):
        c["b"]
    m = Memory(index="a b")
    m.learn({"a": 0})
    with pytest.raises(RuntimeError):
        m.index = "a"
    m.index = "b,a"
    m.reset()
    assert m.index == ("a", "b")
    m.index = "a"
    assert m.index == ("a",)
    m = Memory()
    assert m.index == ()
    m.index = "a b c"
    assert m.index == ("a", "b", "c")
    m = Memory()
    m.index = ("f 18 6 p h 4 3 r t d j 9 10 w x 8 b q s l 14 17 c 0 "
              "n o 1 e i k a 11 15 2 g 5 v 16 12 7 19 u y m z 13")
    assert Memory(index=("x,z,7,16,r,m,p,v,13,l,e,w,o,6,19,8,i,c,q,g,5,n,11,"
                         "12,d,f,y,u,b,0,9,1,3,10,a,17,14,18,4,2,s,15,k,t,j,h"
                         )).index == m.index
    with pytest.raises(ValueError):
        Memory(index="a a")
    m = Memory()
    with pytest.raises(ValueError):
        m.index = "a,b,c,d,e,b,f,g,h"
    entries = [(random.randint(0, 150),
                random.randint(0, 150))
               for _ in range(100_000)]
    random.shuffle(entries)
    keys = list(range(0, 200))
    random.shuffle(keys)
    keys = keys[:10]
    m = Memory()
    def f():
        for d, u in entries:
            m.learn({"d": d, "u": u}, 1)
        start = default_timer()
        for k in keys:
            m.blend("u", {"d": k})
        return default_timer() - start
    no_index = f()
    m.reset()
    m.index = "d"
    assert f() < no_index / 4
    m = Memory(index="d")
    assert f() < no_index / 4

def test_print_chunks(tmp_path):
    m = Memory(index=["d"])
    m.learn({"d": "right", "a": 0.3, "u": 0.5})
    m.learn({"d": "left", "a": 0.5, "u": 0.3})
    m.learn({"d": "right", "a": 0.3, "u": 0.5})
    m.advance()
    m.learn({"d": "right", "a": 0.3, "u": 0.5})
    p = tmp_path / "chunks.csv"
    m.print_chunks(file=p, pretty=False)
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        entries = [s for s in r]
    assert len(entries) == 2
    for line in entries:
        del line["chunk name"]
    assert {'chunk contents': "'a': 0.3, 'd': 'right', 'u': 0.5",
            'chunk created at': '0', 'chunk reference count': '3',
            'chunk references': '0, 0, 1'} in entries
    assert {'chunk contents': "'a': 0.5, 'd': 'left', 'u': 0.3",
            'chunk created at': '0', 'chunk reference count': '1',
            'chunk references': '0'} in entries

def test_salience():
    def sim(x, y):
        return 1 - abs(x - y) / 10

    def deriv(x, y):
        if x == y:
            return 0
        elif x < y:
            return 0.1
        else:
            return -0.1

    def setup_memory(s, d, w=None, **kwd):
        m = Memory(temperature=1, noise=0, **kwd)
        if w:
            m.similarity(["r", "ρ"], s, 1, d)
            m.similarity("h", s, derivative=d, weight=w)
        else:
            m.similarity("r,h,ρ", s, derivative=d)
        for (i, j, k) in [(8, 5, 6), (4, 7, 0), (1, 8, 7), (0, 7, 4), (6, 2, 3), (4, 6, 1), (1, 5, 4), (8, 7, 1),
                          (8, 6, 6), (4, 9, 9), (1, 4, 4), (3, 4, 2), (1, 3, 7), (3, 1, 9), (3, 3, 4), (4, 7, 0),
                          (7, 1, 9), (3, 5, 4), (3, 6, 7), (6, 9, 6), (3, 4, 2), (3, 5, 1), (9, 4, 9), (7, 8, 5),
                          (0, 0, 0), (2, 3, 8), (4, 6, 1), (4, 5, 5), (3, 4, 2), (1, 0, 7), (2, 3, 4), (5, 7, 8)]:
            m.learn({"r": i, "h": j, "ρ": k, "color": "black" if i+j+k % 2 else "gold",
                     "v": i**2 * j, "a": 2*i * (i + 2*j), "m": k * i**2 * j})
            m.advance()
            m.learn({"r": j, "h": k, "ρ": i, "color": "gold" if i+j+k % 2 else "black",
                     "v": j**2 * k, "a": 2*j * (j + 2*k), "m": i * j**2 * k})
            m.advance()
        return m

    m = setup_memory(sim, deriv, mismatch=1)
    assert isclose(m.blend("v", {"color": "black"}), 102.52340070823675)
    assert isclose(m.blend("a", {"color": "gold"}), 149.01302892535938)
    assert isclose(m.blend("m", {"color": "black"}), 546.7513952899268)
    bv, d, ignore = m.blend("v", {"color": "black"}, True)
    assert ignore is None
    assert isclose(bv, 102.52340070823675)
    assert isclose(d[(('r', 4), ('h', 5), ('ρ', 5), ('color', 'black'),
                       ('v', 80), ('a', 112), ('m', 400))],
                   -0.04447050620336284)
    assert isclose(d[(('r', 0), ('h', 7), ('ρ', 4), ('color', 'black'),
                      ('v', 0), ('a', 0), ('m', 0))],
                   -0.08405181892409157)

    bv, ignore, d = m.blend("a", {"r":4.5, "h": 7}, feature_salience=True)
    assert ignore is None
    assert isclose(bv, 142.32864606979274)
    assert isclose(d["r"], -0.8879387514283669)
    assert isclose(d["h"], -0.45996170896264055)

    bv, inst, feat = m.blend("m", {"color": "gold", "r": 3, "ρ": 8.9},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 683.176150736943)
    assert len(inst) == 28
    assert isclose(inst[(('r', 6), ('h', 6), ('ρ', 8), ('color', 'gold'),
                         ('v', 216), ('a', 216), ('m', 1728))], 0.1344501871879563)
    assert isclose(inst[(('r', 0), ('h', 0), ('ρ', 0), ('color', 'gold'),
                         ('v', 0), ('a', 0), ('m', 0))], -0.06770244967281028)
    assert isclose(inst[(('r', 7), ('h', 8), ('ρ', 5), ('color', 'gold'),
                         ('v', 392), ('a', 322), ('m', 1960))], 0.7550582483322396)
    assert isclose(feat["r"], -0.9708704581767997)
    assert isclose(feat["ρ"], -0.23960499460481)

    bv, inst, feat = m.blend("m", instance_salience=True, feature_salience=True)
    assert isclose(bv, 606.8901189653784)
    assert len(inst) == 56
    assert isclose(inst[(('r', 6), ('h', 9), ('ρ', 6), ('color', 'black'),
                         ('v', 324), ('a', 288), ('m', 1944))], 0.12769326117518853)
    assert feat is not None and not feat

    bv, inst, feat = m.blend("v", {"color": "green"},
                             instance_salience=True, feature_salience=True)
    assert bv is None
    assert inst == {}
    assert feat == {}
    bv, inst, feat = m.blend("v", {"color": "green"},
                             instance_salience=True, feature_salience=False)
    assert bv is None
    assert inst == {}
    assert feat is None
    bv, inst, feat = m.blend("v", {"color": "green"},
                             instance_salience=False, feature_salience=True)
    assert bv is None
    assert inst is None
    assert feat == {}

    m.mismatch = 0
    bv, inst, feat = m.blend("m", {"color": "black", "r": 3.14, "ρ": 5.2},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 546.7513952899268)
    assert len(inst) == 28
    assert isclose(inst[(('r', 9), ('h', 4), ('ρ', 9), ('color', 'black'),
                         ('v', 324), ('a', 306), ('m', 2916))], 0.46090494091242534)
    assert isclose(feat["r"], 0)
    assert isclose(feat["ρ"], 0)

    m.mismatch = 2
    bv, inst, feat = m.blend("m", {"color": "gold", "r": 3, "ρ": 8.9},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 711.3277138478871)
    assert len(inst) == 28
    assert isclose(inst[(('r', 6), ('h', 6), ('ρ', 8), ('color', 'gold'),
                         ('v', 216), ('a', 216), ('m', 1728))], 0.18392706129913897)
    assert isclose(feat["r"], -0.9244979288917802)
    assert isclose(feat["ρ"], -0.38118706624806803)

    m.threshold = -2
    bv, inst, feat = m.blend("m", {"color": "gold", "r": 3, "ρ": 8.9},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 928.7459840671638)
    assert len(inst) == 4
    assert isclose(inst[(('r', 4), ('h', 2), ('ρ', 3), ('color', 'gold'),
                         ('v', 32), ('a', 64), ('m', 96))], -0.5204062526719364)
    assert isclose(inst[(('r', 4), ('h', 9), ('ρ', 9), ('color', 'gold'),
                         ('v', 144), ('a', 176), ('m', 1296))], 0.22615988599645534)
    assert isclose(inst[(('r', 3), ('h', 4), ('ρ', 2), ('color', 'gold'),
                         ('v', 36), ('a', 66), ('m', 72))], -0.41623220005499945)
    assert isclose(inst[(('r', 7), ('h', 8), ('ρ', 5), ('color', 'gold'),
                         ('v', 392), ('a', 322), ('m', 1960))], 0.7104785667304807)
    assert isclose(feat["r"], -0.6771428813507683)
    assert isclose(feat["ρ"], -0.7358515599195121)

    m.threshold = -1
    assert m.blend("m", {"color": "gold", "r": 3, "ρ": 8.9},
                   instance_salience=True, feature_salience=True) == (None, {}, {})
    assert m.blend("m", {"color": "gold", "r": 3, "ρ": 8.9},
                   instance_salience=False, feature_salience=True) == (None, None, {})
    assert m.blend("m", {"color": "gold", "r": 3, "ρ": 8.9},
                   instance_salience=True, feature_salience=False) == (None, {}, None)
    assert m.blend("m", {"color": "gold", "r": 3, "ρ": 8.9},
                   instance_salience=False, feature_salience=False) is None
    assert m.blend("m", {"color": "gold", "r": 3, "ρ": 8.9}) is None

    bv, inst, feat = m.blend("m", {"color": "black", "r": 3, "ρ": 8.9},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 1400)
    assert len(inst) == 1
    assert isclose(inst[(('r', 5), ('h', 7), ('ρ', 8), ('color', 'black'),
                         ('v', 175), ('a', 190), ('m', 1400))], 0)
    assert isclose(feat["r"], 0)
    assert isclose(feat["ρ"], 0)

    m = Memory(mismatch=1)
    m.learn({"x": 1})
    m.advance()
    bv, inst, feat = m.blend("x", instance_salience=True, feature_salience=True)
    assert isclose(bv, 1)
    assert len(inst) == 1
    assert isclose(inst[(("x", 1),)], 0)
    assert feat == {}
    for i in range(1000):
        m.learn({"x": 1})
        m.advance()
    assert isclose(bv, 1)
    assert len(inst) == 1
    assert isclose(inst[(("x", 1),)], 0)
    assert feat == {}

    m = setup_memory(sim, deriv, w=3, mismatch=0.7)
    bv, inst, feat = m.blend("m", {"ρ": 5.3},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 641.6940850845056)
    assert len(inst) == 56
    assert isclose(inst[(('r', 7), ('h', 8), ('ρ', 5), ('color', 'black'),
                         ('v', 392), ('a', 322), ('m', 1960))], 0.16384860333473814)
    assert isclose(feat["ρ"], -1.0)

    bv, inst, feat = m.blend("m", {"r": 6, "h": 6, "ρ": 8},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 815.7193117063043)
    assert len(inst) == 56
    assert isclose(inst[(('r', 6), ('h', 6), ('ρ', 8), ('color', 'gold'),
                         ('v', 216), ('a', 216), ('m', 1728))], 0.13832035175108234)
    assert isclose(inst[(('r', 7), ('h', 0), ('ρ', 4), ('color', 'gold'),
                         ('v', 0), ('a', 98), ('m', 0))], -0.051204105261293466)
    assert isclose(feat["r"], -0.5056057256342746)
    assert isclose(feat["h"], -0.8520315588024319)
    assert isclose(feat["ρ"], -0.13566529773872466)

    bv, inst, feat = m.blend("a", {"r": 6, "h": 6, "ρ": 8},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 158.09970728885781)
    assert len(inst) == 56
    assert isclose(inst[(('r', 6), ('h', 6), ('ρ', 8), ('color', 'gold'),
                         ('v', 216), ('a', 216), ('m', 1728))], 0.07236318010571169)
    assert isclose(inst[(('r', 7), ('h', 0), ('ρ', 4), ('color', 'gold'),
                         ('v', 0), ('a', 98), ('m', 0))], -0.031096798268279532)
    assert isclose(inst[(('r', 3), ('h', 6), ('ρ', 7), ('color', 'black'),
                         ('v', 54), ('a', 90), ('m', 378))], -0.0833392501594482)
    assert isclose(feat["r"], -0.4962203932830241)
    assert isclose(feat["h"], -0.8674012821292578)
    assert isclose(feat["ρ"], -0.037152887513091704)

    m = setup_memory(True, deriv, mismatch=1)
    bv, inst, feat = m.blend("a", {"r": 6, "h": 6, "ρ": 8, "color": "black"},
                             instance_salience=True, feature_salience=True)
    assert isclose(bv, 130.6814457919905)
    assert len(inst) == 28
    assert isclose(inst[(('r', 4), ('h', 6), ('ρ', 1), ('color', 'black'),
                         ('v', 96), ('a', 128), ('m', 96))], -0.014343113569285259)
    assert isclose(feat["r"], -0.6577129862176535)
    assert isclose(feat["h"], -0.741625596990525)
    assert isclose(feat["ρ"], -0.13192839591651323)

    m = setup_memory(sim, None, mismatch=1)
    bv, inst, feat = m.blend("a", {"r": 6, "h": 6}, instance_salience=True)
    assert feat is None
    assert isclose(bv, 149.48556815120594)
    assert len(inst) == 56
    assert isclose(inst[(('r', 4), ('h', 5), ('ρ', 5), ('color', 'black'),
                         ('v', 80), ('a', 112), ('m', 400))], -0.049933984477157314)
    with pytest.raises(RuntimeError):
        m.blend("a", {"r": 6, "h": 6}, feature_salience=True)
    with pytest.raises(RuntimeError):
        m.blend("a", {"r": 6, "h": 6}, True, True)
