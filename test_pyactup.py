# Copyright 2018-2022 Carnegie Mellon University

from pyactup import *
import pyactup

import math
import numpy as np
import pickle
import pytest
import random
import sys

from math import isclose
from pprint import pprint
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
    m.similarity(["b"], True)
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
    m.similarity(["a"], weight=2)
    m.similarity(["b"], weight=10)
    test_one(4, 3, 0, -0.50, 2, 0, -10, 0)
    m.similarity(["b"])
    assert m._similarities.get("b") is None
    m.use_actr_similarity = True
    m.similarity(["b"], weight=20)
    m.similarity(["a"], lambda x, y: sim(x, y) - 1, 4)
    test_one(3, 4, 0, -1, 2, 0, -20, 0)

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

def test_blend():
    for m in [Memory(temperature=1, noise=0),
              Memory(temperature=1, noise=0, index="a"),
              Memory(temperature=1, noise=0, index="b"),
              Memory(temperature=1, noise=0, index="a,b")]:
        m.learn({"a":1, "b":1})
        m.learn({"a":2, "b":2})
        m.advance()
        print(m)
        print(m.blend("b", {"a":1}))
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
        assert isclose(dict(p)[3], 0.1526351150445461)
        b, p = m.discrete_blend("o")
        assert b == 5
        assert isclose(dict(p)[4], 0.07519141419785559)

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
        assert m.chunks[0].references == [3]
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
        assert m.chunks[0].references == [0]
        m.learn({"n":2})
        m.advance()
        assert len(m.chunks) == 2
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 2
        assert m.chunks[0].reference_count == 2
        assert m.chunks[0].references == [0, 2]
        assert m.chunks[1].reference_count == 1
        assert m.chunks[1].references == [1]
        m.reset()
        m.optimized_learning = True
        assert len(m.chunks) == 0
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 1
        assert m.chunks[0].references == []
        m.learn({"n":2})
        m.advance()
        assert len(m.chunks) == 2
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 2
        assert m.chunks[0].reference_count == 2
        assert m.chunks[0].references == []
        assert m.chunks[1].reference_count == 1
        assert m.chunks[1].references == []
        m.reset()
        m.optimized_learning = 1
        assert len(m.chunks) == 0
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 1
        assert m.chunks[0].references == [0]
        m.learn({"n":2})
        m.advance()
        assert len(m.chunks) == 2
        m.learn({"n":1})
        m.advance()
        assert len(m.chunks) == 2
        assert m.chunks[0].reference_count == 2
        assert m.chunks[0].references == [2]
        assert m.chunks[1].reference_count == 1
        assert m.chunks[1].references == [1]
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
        assert m.chunks[2].references == [3]
        assert m.chunks[0].references == [0, 0, 1, 3, 4, 4, 5, 6, 7]
        assert m.chunks[1].references == [2, 8]
        f(True)
        assert m.chunks[0].references == []
        assert m.chunks[1].references == []
        assert m.chunks[2].references == []
        f(5)
        assert m.chunks[0].references == [4, 4, 5, 6, 7]
        assert m.chunks[1].references == [2, 8]
        assert m.chunks[2].references == [3]
        f(4)
        assert m.chunks[0].references == [4, 5, 6, 7]
        assert m.chunks[1].references == [2, 8]
        assert m.chunks[2].references == [3]
        f(2)
        assert m.chunks[0].references == [6, 7]
        assert m.chunks[1].references == [2, 8]
        assert m.chunks[2].references == [3]
        f(1)
        assert m.chunks[0].references == [7]
        assert m.chunks[1].references == [8]
        assert m.chunks[2].references == [3]

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
    entries = [(random.randint(0, 200),
                random.randint(0, 200))
               for _ in range(1_000_000)]
    random.shuffle(entries)
    keys = list(range(0, 200))
    random.shuffle(keys)
    keys = keys[:50]
    m = Memory()
    for d, u in entries:
        m.learn({"d": d, "u": u}, 1)
    start = default_timer()
    for k in keys:
        m.blend("u", {"d": k})
    no_index = default_timer() - start
    m = Memory(index="d")
    for d, u in entries:
        m.learn({"d": d, "u": u}, 1)
    start = default_timer()
    for k in keys:
        m.blend("u", {"d": k})
    with_index = default_timer() - start
    assert with_index < no_index / 4
