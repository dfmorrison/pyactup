.PHONY: dist clean upload

dist:	clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist/*

upload: dist
	twine upload -u dfmorrison dist/*
	scp -r doc/_build/html/* dfm@koalemos.psy.cmu.edu:/var/www/html/pyactup/
