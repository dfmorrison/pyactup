.PHONY: dist clean doc upload

dist:	clean doc
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist/*

doc:
	cd doc/ ; make html

upload: dist
	twine upload -u __token__ dist/*
