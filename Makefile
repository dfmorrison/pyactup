.PHONY: dist clean upload

dist:	clean
	python setup.py sdist bdist_wheel
	cd doc/ ; make html

clean:
	rm -rf dist/*

upload: dist
	twine upload -u __token__ dist/*
	scp -r doc/_build/html/* dfm@koalemos.psy.cmu.edu:/var/www/html/pyactup/
