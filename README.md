PyACTUp is a lightweight Python implementation of a subset of the
ACT-R cognitive architecture’s Declarative Memory, suitable for
incorporating into other Python models and applications. It is
inspired by the ACT-UP cognitive modeling toolbox.

There is [online documentation of PyACTUp](http://halle.psy.cmu.edu/pyactup/),
and the [sources](https://bitbucket.org/dfmorrison/pyactup/) are on Bitbucket.

PyACTUp requires Python version 3.6 or later. PyACTUp also works in
recent versions of PyPy.

Normally, assuming you are connected to the internet, to install
PyACTUp you should simply have to type at the command line

    pip install pyactup

Depending upon various possible variations in how Python and your
machine are configured you may have to modify the above in various
ways

* you may need to ensure your virtual environment is activated
* you may need use an alternative scheme your Python IDE supports
* you may need to call it `pip3` instead of simply `pip`
* you may need to precede the call to `pip` by `sudo`
* you may need to use some combination of the above

If you are unable to install PyACTUp as above, you can instead
download a tarball from
[bitbucket](https://bitbucket.org/dfmorrison/pyactup/downloads/).
The tarball will have a filename something like `pyactup-1.1.1.tar.gz`.
Assuming this file is at `/some/directory/pyactup-1.1.1.tar.gz` install
it by typing at the command line

    pip install /some/directory/pyactup-1.1.1.tar.gz

Alternatively you can untar the tarball with

    tar -xf /some/directory/pyactup-1.1.1.tar.gz

and then change to the resulting directory and type

    python setup.py install

PyACTUp is released under the following MIT style license:

Copyright (c) 2018-2021 Carnegie Mellon University

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
