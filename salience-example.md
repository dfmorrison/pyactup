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

It’s not clear how valuable the ground truth is here, since blending is so strongly influenced by the
distribution of experiences, as well as their frequencies and recencies, and the details of any
similarity functions used, but in the above $v$ is the volume of a circular cylinder of radius $r$
and height $h$, which is $r^2h$.

We will compute a blended value of $v$ for $r=2$ and $h=2$. To do this we will use a slightly more
interesting similarity function than the usual linear one, $\xi(x,y)=1-\sqrt{\frac{|x-y|}{max(x,y)}}$.
We will assign $\xi$ as the similarity function for both $r$ and $h$.

    def sim(x, y):
        return 1 - math.sqrt(abs(x - y) / max(x, y))
    m.similarity(["r", "h"], sim)

Recall that when we defined our PyACTUp memory, we set the noise to zero, and both the blending temperature
and the mismatch penalty to unity; the decay parameter retains its default value of $0.5$. With these parameters
we now compute the blended value of $v$ for $r=2$ and $h=2$, and print the probabilities of retrieval for the
four chunks, as well as the resulting blended value.

    m.activation_history = True
    bv = m.blend("v", {"r": 2, "h": 2})
    for d in m.activation_history:
        slots = dict(d["attributes"])
        print(f'p={d["retrieval_probability"]:.2}, r={slots["r"]}, h={slots["h"]}, v={slots["v"]}')
    print(f"BV={bv:.2}")
    p=0.43, r=1, h=1, v=1
    p=0.15, r=3, h=3, v=27
    p=0.14, r=1, h=3, v=3
    p=0.29, r=3, h=1, v=9
    BV=7.4

To compute the saliencies of $r$ and $h$ we need the derivative of the similarity function

$$
\frac{\partial \xi}{\partial x}(x,y) = \left\{
\matrix{
\frac{y}{2x^2\sqrt{1-\frac{y}{x}}} \text{ if $x>y$}\\
f(x,y)  \text{ if $y>x$}\\
\text{undefined if $x=y$}}
\right.
$$
