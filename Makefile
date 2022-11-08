.PHONY: dist clean upload

dist:	clean
	python -m build -n

clean:
	rm -rf dist/*

upload: dist
	twine upload -u dfmorrison dist/*
