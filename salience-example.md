### A Simple Example of Salience for ACT-UP

> We think in generalities, but we live in detail. – Alfred North Whitehead

This is a simple example of salience computation, just to make sure I understand how it works before adding
it to PyACTUp.
We use PyACTUp (albeit called from Lisp) to compute the relevant activations and probabilities of retrieval.
First we create a small ensemble of chunks, each with three slots, $r$, $h$ and $v$, each holding
a real number; further, $r$ and $h$ should be less than or equal to 16. While we have six experiences,
there are only four chunks, because of repetition at different times.

    (import-module "pyactup")

    (defparameter *m* (pyactup:memory :temperature 1 :noise 0 :mismatch 1))

    (iter (for (r h v) :in '((1 1 1) (3 3 27) (1 3 3) (1 1 1) (1 1 1) (3 1 9)))
          (chain *m* (learn (plist-hash-table `("r" ,r "h" ,h "v" ,v))))
          (chain *m* (advance)))
    (uiop:with-temporary-file (:pathname p)
      (chain *m* (print_chunks (namestring p)))
      (format t "~&~A~%" (uiop:read-file-string p)))

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

We will compute a blended value of $v$ for new values of $r$ and $h$. To do this we will use a slightly more
interesting similarity function than the usual linear one, $\xi(x,y)=1-\sqrt{\frac{|x-y|}{16}}$.
We use this similarity function not from any principled belief it is somehow “right,” but simply to have
something to play with that doesn’t have the degenerate derivatives that the common linear similarly
function does.
We will assign $\xi$ as the similarity function for both $r$ and $h$.

    (export-function (lambda (x y)
                       (- 1 (sqrt (/ (abs (- x y)) 16.0d0))))
                     "sim")
    (chain *m* (similarity '("r" "h") (python-eval "sim")))

Recall that when we defined our PyACTUp memory, we set the noise to zero, and both the blending temperature
and the mismatch penalty to unity; the decay parameter retains its default value of $0.5$. With these parameters
we now compute the blended value of $v$ for $r=2$ and $h=2$, and print the probabilities of retrieval.

    (defvar *data*)
    (setf (chain *m* activation_history) t)
    (let ((bv (chain *m* (blend "v" (plist-hash-table '("r" 2 "h" 2))))))
      (setf *data* (iter (for tab :in-vector (chain *m* activation_history))
                         (for attrs := (coerce (gethash "attributes" tab) 'list))
                         (collect `(:retrieval-probability ,(gethash "retrieval_probability" tab)
                                    ,@(iter (for k :in '(:r :h :v))
                                            (nconcing `(,k ,(cadr (assoc k attrs
                                                                         :test #'string-equal)))))))))
      (format t "~&~:W~2%BV = ~A~%" *data* bv))

    ((:RETRIEVAL-PROBABILITY 0.46503928 :R 1 :H 1 :V 1)
     (:RETRIEVAL-PROBABILITY 0.12286361 :R 3 :H 3 :V 27)
     (:RETRIEVAL-PROBABILITY 0.1373657 :R 1 :H 3 :V 3)
     (:RETRIEVAL-PROBABILITY 0.2747314 :R 3 :H 1 :V 9))

    BV = 6.6670365

To compute the saliencies of $r$ and $h$ we need the derivative of the similarity function, which is

```math
\frac{\partial}{\partial x}\xi(x,y) = \left\{\matrix{\frac{1}{8 \sqrt{x-y}} \ \ \ \ \ \ \ \ \text{ if $x>y$}\\ \frac{-1}{8 \sqrt{y-x}} \ \ \ \ \ \ \ \ \text{ if $x < y$}\\ \text{undefined   if $x=y$}} \right.
```

We can now compute the saliences using equation (7) from ACT-R-saliency-computations-v6.pdf.

    (defun deriv (x y)
      (labels ((result (diff sign)
                 (/ sign (* 8 (sqrt diff)))))
        ;; returns nil if x == y
        (cond ((> x y) (result (- x y) 1))
              ((< x y) (result (- y x) -1)))))

    (defun salience (attr target)
      (labels ((weighted-sum (values)
                 (iter (for x :in *data*)
                       (for v :in values)
                       (sum (* (getf x :retrieval-probability) v)))))
        (let* ((derivs (iter (for x :in *data*)
                             (collect (deriv (getf x attr) target))))
               (sums (weighted-sum derivs)))
          (weighted-sum (iter (for x :in *data*)
                              (for d :in derivs)
                              (collect (* (getf x :v) (- d sums))))))))

    (format t "~&r salience = ~A, h salience = ~A~%"
            (salience :r 2) (salience :h 2))

    r salience = 0.7847799, h salience = 0.49861407

So, have I got this right?

Now a few further questions.

#### PyACTUp attributes can take different similarity functions and weights

Different attributes can use completely different similarity functions in PyACTUp.
I presume this is no problem, we just use the derivative of the correct function
for each attribute. Is this correct?

A little more complicated is that these different attributes can also have weights
applied to their similarity function values. This was something requested by
Sterling, which has already proved useful in a project with Coty and a group at Aptima.
Essentially the mismatch penalty is multiplied by the weight before being applied to
the attributes similarity value; if not explicitly supplied, weights default to unity,
giving the usual behavior. How do we work this into equation (7)?

#### Caching

Another feature added at Sterling’s request is caching of similarity values. This depends
upon the two quite reasonable constraints PyACTUp demans of similarity functions

- that they actually *be* functions, in the sense that whenever called with the same
  argument values they always return the same function value

- and that they be commutative in their arguments, that is $\xi(y,x) \equiv \xi(x,y)$.

I see no reason that we can’t cache the values of the derivatives of the similarity functions, too, though
the second consideration above no longer applies to the partial derivative..
Does anyone else see a problem with this? Whether or not it is valuable is an open question;
I’m leaning towards doing it just to be symmetrical with the similarity function caching behavior,
but don’t feel particularly strongly one way or the other.

#### Undefined derivative values

For the sorts of similarity functions we use it will be a common occurrence that they will not
be differentiable over their entire domains; in particular, there is frequently a singularity
when their two arguments are equal.

To date we’ve dealt with this with a wave of the hand, mumble something about “arbitrary.” But
to build it, we need to know what to do in this case.
